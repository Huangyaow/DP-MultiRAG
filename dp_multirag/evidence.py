"""Module 2 — Controlled Evidence Utilization.

For a single turn the module runs:

    (1) Evidence DP transformation
            C̃_t = M_trans( C_t ; ε_t^trans, δ_t^trans )
        with three families of operators:
            * Redaction      — HIGH-tier spans                 → [REDACTED]
            * Abstraction    — LOW-tier spans                  → coarse type
            * Gaussian noise — numeric tokens                  → x + N(0, σ²)
        Gaussian noise scale follows the analytical mechanism
            σ_t^trans = Δ_trans · √(2 ln(1.25 / δ)) / ε_t^trans .

    (2) Sufficiency r_t^suf = f_suf(q̃_t, C̃_t, L_{t-1}).

    (3) Three-way decision
            a_t = Generate   if r_t^suf ≥ τ_s
                  Retrieve   if r_t^suf <  τ_s  AND   B_t > τ_b
                  Abstain    otherwise.
"""

from __future__ import annotations

import math
import random
import re
from typing import List, Tuple

from . import ledger as L


NUMERIC_RANGE = (0.0, 10000.0)
DELTA_TRANS = NUMERIC_RANGE[1] - NUMERIC_RANGE[0]


def gaussian_sigma(eps: float, delta: float, sensitivity: float) -> float:
    """``σ = Δ · √(2 ln(1.25 / δ)) / ε`` (analytical Gaussian mechanism)."""
    if eps <= 0:
        return float("inf")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / max(delta, 1e-12))) / eps


_NUM_RE = re.compile(r"\b\d+(?:[\.\-]\d+)*\b")


def _abstract(span_text: str) -> str:
    """Replace a sensitive span with a coarse type tag."""
    s = span_text.strip()
    if _NUM_RE.search(s) and any(c.isdigit() for c in s):
        return "[NUMERIC_RANGE]"
    if "@" in s:
        return "[EMAIL]"
    if any(t in s.lower() for t in ("st.", "street", "avenue", "ave.", "road",
                                    "branch", "office")):
        return "[LOCATION]"
    if any(t in s.lower() for t in ("am", "pm", "monday", "tuesday",
                                    "wednesday", "thursday", "friday")):
        return "[TIME]"
    if s.split()[0][:1].isupper():
        return "[NAME]"
    return "[ABSTRACTED]"


def _add_gaussian_to_numbers(text: str, sigma: float, rng: random.Random
                             ) -> Tuple[str, int]:
    """Add Gaussian noise to numeric tokens (clipped to ``NUMERIC_RANGE``)."""
    n_perturbed = 0

    def _repl(m):
        nonlocal n_perturbed
        try:
            x = float(m.group(0).split("-")[0])
        except ValueError:
            return m.group(0)
        x = max(NUMERIC_RANGE[0], min(NUMERIC_RANGE[1], x))
        x_noisy = x + rng.gauss(0.0, sigma)
        n_perturbed += 1
        return f"~{int(round(x_noisy))}"

    return _NUM_RE.sub(_repl, text), n_perturbed


def transform_evidence(cands: list, ledger: dict, cfg: dict,
                       risk_per_doc: List[float], rng: random.Random) -> dict:
    """Apply ``M_trans`` to each retrieved document.

    The per-document operator is selected by Φ:

      * ``Φ ≥ phi_high``           — Redaction of HIGH-tier spans
                                      + abstraction of LOW-tier spans
      * ``phi_low ≤ Φ < phi_high`` — Abstraction of HIGH-tier spans
                                      + Gaussian noise on numeric tokens
      * ``Φ < phi_low``            — Identity (no transformation)
    """
    eps_trans_t = cfg["eps_t_trans"]
    delta = ledger["config"]["delta_global"]
    sigma = gaussian_sigma(eps_trans_t, delta, DELTA_TRANS)

    transformed = []
    for d, ph in zip(cands, risk_per_doc):
        text = d["text"]
        ops_used = []
        n_redacted = n_abstracted = n_noised = 0

        if ph >= cfg["phi_high"]:
            for s in d.get("sensitive_spans", []):
                if s["text"] in text:
                    if s["tier"] == "high":
                        text = text.replace(s["text"], "[REDACTED]")
                        n_redacted += 1
                    else:
                        text = text.replace(s["text"], _abstract(s["text"]))
                        n_abstracted += 1
            ops_used.append("redact+abstract")
        elif ph >= cfg["phi_low"]:
            for s in d.get("sensitive_spans", []):
                if s["text"] in text and s["tier"] == "high":
                    text = text.replace(s["text"], _abstract(s["text"]))
                    n_abstracted += 1
            text, n_noised = _add_gaussian_to_numbers(text, sigma, rng)
            ops_used.append("abstract+gauss")
        else:
            ops_used.append("identity")

        cost = (
            0.0 if "identity" in ops_used
            else eps_trans_t * (0.6 + 0.4 * ph)
        )
        if cost > 0:
            L.charge_transform(ledger, d["id"], cost)

        transformed.append({
            "id": d["id"],
            "text": text,
            "ops": ops_used,
            "n_redacted": n_redacted,
            "n_abstracted": n_abstracted,
            "n_noised": n_noised,
            "delta_eps_trans": cost,
            "phi": ph,
        })

    return {
        "evidence": transformed,
        "sigma": sigma,
        "delta": delta,
    }


def sufficiency(q_tilde: str, transformed: list, ledger: dict) -> float:
    """``r_t^suf = f_suf(q̃_t, C̃_t, L_{t-1})`` in ``[0, 1]``.

    Higher when post-transform evidence still contains content tokens
    overlapping with the rewritten query and the ledger has plenty of
    budget left; lower when most sensitive spans have been masked or the
    candidate documents are close to their ε caps.
    """
    if not transformed:
        return 0.0

    q_toks = set(re.findall(r"[a-zA-Z0-9]+", q_tilde.lower()))
    q_toks = {t for t in q_toks if len(t) > 2}

    overlap, mask_penalty, budget_health = 0.0, 0.0, 0.0
    for e in transformed:
        d_toks = set(re.findall(r"[a-zA-Z0-9]+", e["text"].lower()))
        overlap += len(q_toks & d_toks) / max(1, len(q_toks))
        mask_penalty += 0.5 * (e["n_redacted"] > 0) + 0.25 * (e["n_abstracted"] > 0)
        budget_health += min(1.0, L.remaining(ledger, e["id"], "all")
                             / max(1e-6, ledger["config"]["eps_global"]))

    n = len(transformed)
    overlap /= n
    mask_penalty = min(1.0, mask_penalty / n)
    budget_health /= n

    return max(0.0, min(1.0,
                        0.55 * overlap
                        + 0.25 * budget_health
                        - 0.30 * mask_penalty
                        + 0.30))


def decide_action(r_suf: float, ledger: dict, selected_ids: list,
                  tau_s: float, tau_b: float) -> str:
    """Three-way decision ``a_t ∈ {"generate", "retrieve", "abstain"}``."""
    if r_suf >= tau_s:
        return "generate"

    if not selected_ids:
        return "abstain"
    B_t = min(L.remaining(ledger, did, "all") for did in selected_ids)
    if B_t > tau_b:
        return "retrieve"
    return "abstain"
