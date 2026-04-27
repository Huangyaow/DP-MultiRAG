"""Module 4 — Document-level Privacy Ledger.

Implements the cross-turn privacy book-keeping defined in DP-MultiRAG.
Per-document state is updated after every turn:

    L_i^(t) = f( L_i^(t-1), e_i^(t), Δε_i^(t) )

with cumulative exposure

    Exp(d_i) = α_1 · n_i^ret + α_2 · n_i^ctx + α_3 · n_i^align

and three-component privacy consumption

    ε_i^(1:T) = Σ_t ( Δε_i^{ret,t} + Δε_i^{trans,t} + Δε_i^{gen,t} )

Hierarchical budget structure

    ε_global  ≥  Σ_t ε_t  =  Σ_t ( ε_t^ret + ε_t^trans + ε_t^gen )

is enforced by the helper :func:`global_remaining`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def init_ledger(
    docs: List[dict],
    eps_global: float,
    delta_global: float,
    split: Tuple[float, float, float] = (0.30, 0.20, 0.50),
    alpha: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> dict:
    """Allocate per-document budgets and zero exposure counters.

    Args:
        docs: knowledge-base documents (only ``id`` is used here).
        eps_global: total per-document privacy budget ε_global.
        delta_global: failure parameter δ used by the Gaussian mechanisms.
        split: fraction of ε_global allocated to (ret, trans, gen).
        alpha: weights (α_1, α_2, α_3) for the cumulative exposure formula.

    Returns:
        A dict with two keys: ``"docs"`` mapping document id to per-document
        record, and ``"config"`` carrying the global parameters.
    """
    if abs(sum(split) - 1.0) > 1e-6 and sum(split) > 1.0:
        raise ValueError(f"epsilon split must sum to <= 1.0, got {sum(split)}")

    eps_ret_cap = eps_global * split[0]
    eps_trans_cap = eps_global * split[1]
    eps_gen_cap = eps_global * split[2]

    rows = {}
    for d in docs:
        rows[d["id"]] = {
            "n_ret": 0,
            "n_ctx": 0,
            "n_align": 0,
            "exposure": 0.0,
            "eps_ret_used": 0.0,
            "eps_trans_used": 0.0,
            "eps_gen_used": 0.0,
            "eps_ret_cap": eps_ret_cap,
            "eps_trans_cap": eps_trans_cap,
            "eps_gen_cap": eps_gen_cap,
        }

    return {
        "docs": rows,
        "config": {
            "eps_global": eps_global,
            "delta_global": delta_global,
            "split": split,
            "alpha": alpha,
        },
    }


def exposure(ledger: dict, doc_id: str) -> float:
    """Return ``Exp(d_i) = α_1 n_i^ret + α_2 n_i^ctx + α_3 n_i^align``."""
    a1, a2, a3 = ledger["config"]["alpha"]
    r = ledger["docs"][doc_id]
    return a1 * r["n_ret"] + a2 * r["n_ctx"] + a3 * r["n_align"]


def total_used(ledger: dict, doc_id: str) -> float:
    """Cumulative ε spent on ``doc_id`` across ret/trans/gen."""
    r = ledger["docs"][doc_id]
    return r["eps_ret_used"] + r["eps_trans_used"] + r["eps_gen_used"]


def remaining(ledger: dict, doc_id: str, kind: str = "all") -> float:
    """Per-document remaining budget for ``kind`` ∈ {ret, trans, gen, all}."""
    r = ledger["docs"][doc_id]
    if kind == "ret":
        return max(0.0, r["eps_ret_cap"] - r["eps_ret_used"])
    if kind == "trans":
        return max(0.0, r["eps_trans_cap"] - r["eps_trans_used"])
    if kind == "gen":
        return max(0.0, r["eps_gen_cap"] - r["eps_gen_used"])
    if kind == "all":
        return max(0.0, ledger["config"]["eps_global"] - total_used(ledger, doc_id))
    raise ValueError(f"unknown kind={kind}")


def global_remaining(ledger: dict) -> float:
    """``ε_global − max_i ε_i^(1:T)`` (worst-case per-document residual)."""
    if not ledger["docs"]:
        return ledger["config"]["eps_global"]
    return ledger["config"]["eps_global"] - max(
        total_used(ledger, did) for did in ledger["docs"]
    )


def charge_retrieval(ledger: dict, doc_id: str, delta_eps: float) -> None:
    """Apply ``Δε_i^{ret,t}`` (Module 1)."""
    r = ledger["docs"][doc_id]
    r["n_ret"] += 1
    r["eps_ret_used"] = min(r["eps_ret_cap"], r["eps_ret_used"] + delta_eps)
    r["exposure"] = exposure(ledger, doc_id)


def charge_transform(ledger: dict, doc_id: str, delta_eps: float) -> None:
    """Apply ``Δε_i^{trans,t}`` (Module 2)."""
    r = ledger["docs"][doc_id]
    r["n_ctx"] += 1
    r["eps_trans_used"] = min(r["eps_trans_cap"], r["eps_trans_used"] + delta_eps)
    r["exposure"] = exposure(ledger, doc_id)


def charge_generation(ledger: dict, doc_id: str, delta_eps: float) -> None:
    """Apply ``Δε_i^{gen,t}`` (Module 3)."""
    r = ledger["docs"][doc_id]
    r["n_align"] += 1
    r["eps_gen_used"] = min(r["eps_gen_cap"], r["eps_gen_used"] + delta_eps)
    r["exposure"] = exposure(ledger, doc_id)


def decay_exposure(ledger: dict, decay: float) -> None:
    """Soft cross-turn decay on the cached exposure value.

    Counters ``n_i^*`` are not decayed; only the weighted exposure value
    used inside ``Φ(d_i)`` is multiplied by ``decay`` so that recent leaks
    weigh more than older ones.
    """
    for row in ledger["docs"].values():
        row["exposure"] *= decay


def snapshot(ledger: dict, ids: Iterable[str] | None = None) -> str:
    """Compact textual snapshot for logging."""
    rows = ledger["docs"]
    keys = list(ids) if ids is not None else list(rows.keys())
    parts = []
    for did in keys:
        r = rows[did]
        parts.append(
            f"{did}: E={r['exposure']:.2f} "
            f"ε(r/t/g)={r['eps_ret_used']:.2f}/"
            f"{r['eps_trans_used']:.2f}/"
            f"{r['eps_gen_used']:.2f}"
        )
    return " | ".join(parts)


def full_dump(ledger: dict) -> Dict[str, dict]:
    """Return a serialisable dump of the full ledger."""
    out = {}
    for did, r in ledger["docs"].items():
        out[did] = {
            **r,
            "exposure_recomputed": exposure(ledger, did),
            "eps_used_total": total_used(ledger, did),
            "eps_remaining_global": remaining(ledger, did, "all"),
        }
    return out
