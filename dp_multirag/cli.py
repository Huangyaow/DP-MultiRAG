"""Command-line entry point for DP-MultiRAG.

Usage:
    python -m dp_multirag.cli --eps 1.0 --top-k 10
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Optional

from .config import build_parser
from .llm import LLMClient
from .pipeline import per_turn_budget, run_turn
from . import ledger as L


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


if _color_enabled():
    C_DIM = "\033[2m"; C_END = "\033[0m"
    C_CYAN = "\033[36m"; C_YEL = "\033[33m"
    C_GRN = "\033[32m"; C_RED = "\033[31m"; C_MAG = "\033[35m"; C_BLU = "\033[34m"
else:
    C_DIM = C_END = C_CYAN = C_YEL = C_GRN = C_RED = C_MAG = C_BLU = ""


def banner(title: str) -> None:
    bar = "=" * 72
    print(f"\n{C_CYAN}{bar}\n{title}\n{bar}{C_END}")


def step(tag: str, content: str, color: str = C_YEL) -> None:
    print(f"{color}[{tag}]{C_END} {content}")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_turn(t: int, n_turns: int, raw_q: str, ret_cfg: dict, result,
                used_reference: bool) -> None:
    """Render a single :class:`TurnResult` to stdout."""
    banner(
        f"Round {t}/{n_turns}   "
        f"ε_t^ret={ret_cfg['eps_t_ret']:.3f}  "
        f"ε_t^trans={ret_cfg['eps_t_trans']:.3f}  "
        f"ε_t^gen={ret_cfg['eps_t_gen']:.3f}"
    )
    print(f"{C_MAG}[User]{C_END} {raw_q}")
    step("M1 Rewrite", result.q_tilde)
    step("M1 Score+Φ",
         "  ".join(f"{i}:Score={s:+.2f}(Rel={r:.2f},Φ={p:.2f})"
                   for i, s, r, p in zip(result.selected_ids,
                                          result.scores,
                                          result.rel, result.phi)))
    step("M1 GumbelTopK", f"selected={result.selected_ids}", color=C_BLU)
    step("M1 Δε^ret",
         "  ".join(f"{i}:{d:.3f}" for i, d in
                   zip(result.selected_ids, result.delta_eps_ret)),
         color=C_DIM)

    step("M2 σ_trans", f"{result.sigma_trans:.3f}", color=C_DIM)
    for e in result.transformed:
        ops = "/".join(e["ops"])
        preview = e["text"][:160] + ("…" if len(e["text"]) > 160 else "")
        color = C_GRN if "redact" in ops else (C_BLU if "abstract" in ops
                                                else C_DIM)
        step(f"M2 [{e['id']}:{ops}  Δε^trans={e['delta_eps_trans']:.3f}]",
             preview, color=color)

    sufficiency_color = (C_GRN if result.action == "generate"
                         else (C_YEL if result.action == "retrieve"
                               else C_RED))
    if result.re_retrieved:
        step("M2 Re-retrieve",
             "previous attempt insufficient; re-ran Module 1 with widened K",
             color=C_YEL)
    step("M2 Sufficiency",
         f"r_suf={result.sufficiency:.2f}  action={result.action}",
         color=sufficiency_color)

    if result.action == "abstain":
        step("M3 Answer", result.answer, color=C_RED)
    else:
        if used_reference:
            step("M3 mode", "reference-answer decode", color=C_DIM)
        step("M3 σ_gen(τ)",
             f"min={min(result.sigma_gen_per_token, default=0):.2f} "
             f"max={max(result.sigma_gen_per_token, default=0):.2f} "
             f"n_tok={result.n_tokens}",
             color=C_DIM)
        step("M3 Δε^gen",
             "  ".join(f"{i}:{v:.3f}" for i, v in
                       result.delta_eps_gen_per_doc.items()) or "(none)",
             color=C_DIM)
        ans_color = C_RED if not result.delta_eps_gen_per_doc else C_GRN
        step("M3 Answer", result.answer, color=ans_color)

    step("M4 Ledger", "(see Final Ledger at the end)", color=C_DIM)


def render_final(ledger: dict, history: list) -> None:
    banner("Final Ledger (per-document privacy accounting)")
    for did, row in L.full_dump(ledger).items():
        print(f"  {did}")
        for k in ("n_ret", "n_ctx", "n_align",
                  "exposure_recomputed",
                  "eps_ret_used", "eps_trans_used", "eps_gen_used",
                  "eps_used_total", "eps_remaining_global"):
            v = row[k]
            if isinstance(v, float):
                print(f"    {k:>22}: {v:.3f}")
            else:
                print(f"    {k:>22}: {v}")

    banner("Conversation Trace")
    for i, (q, a) in enumerate(history, start=1):
        print(f"  Round {i}")
        print(f"    Q: {q}")
        print(f"    A: {a}")


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)

    llm = LLMClient(args.base_url, args.api_key, args.model)

    docs = load_json(args.docs)
    turns = load_json(args.turns)

    reference_answers = None
    if args.reference_decode:
        reference_answers = load_json(args.reference_answers)
        if len(reference_answers) < len(turns):
            raise ValueError(
                f"--reference-decode needs at least {len(turns)} entries, "
                f"got {len(reference_answers)} in {args.reference_answers}"
            )

    ledger = L.init_ledger(
        docs,
        eps_global=args.eps,
        delta_global=args.delta,
        split=(args.eps_ret_frac, args.eps_trans_frac, args.eps_gen_frac),
        alpha=(args.alpha1, args.alpha2, args.alpha3),
    )

    budgets_t = per_turn_budget(
        eps_global=args.eps,
        eps_ret_frac=args.eps_ret_frac,
        eps_trans_frac=args.eps_trans_frac,
        eps_gen_frac=args.eps_gen_frac,
        n_turns=len(turns),
    )

    cfg = {
        **budgets_t,
        "top_k": args.top_k,
        "gamma": args.gamma,
        "beta": args.beta,
        "w": (args.w1, args.w2, args.w3),
        "alpha": (args.alpha1, args.alpha2, args.alpha3),
        "phi_low": args.phi_low,
        "phi_high": args.phi_high,
        "tau_s": args.tau_s,
        "tau_b": args.tau_b,
    }

    history = []
    for t, raw_q in enumerate(turns, start=1):
        if t > 1:
            L.decay_exposure(ledger, args.decay)

        ref_ans = ref_disc = None
        used_reference = False
        if reference_answers is not None:
            entry = reference_answers[t - 1]
            ref_ans = entry["answer"]
            ref_disc = float(entry.get("disclosure", 1.0))
            used_reference = True

        result = run_turn(
            llm=llm,
            raw_q=raw_q,
            history=history,
            docs=docs,
            ledger=ledger,
            cfg=cfg,
            rng=rng,
            reference_answer=ref_ans,
            reference_disclosure=ref_disc if ref_disc is not None else 1.0,
            re_retrieve_factor=args.re_retrieve_factor,
        )
        render_turn(t, len(turns), raw_q, budgets_t, result, used_reference)
        step("M4 Ledger", L.snapshot(ledger, result.selected_ids), color=C_DIM)

        history.append((raw_q, result.answer))

    render_final(ledger, history)


if __name__ == "__main__":
    main()
