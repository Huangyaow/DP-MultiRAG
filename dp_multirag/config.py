"""Argument parser for :mod:`dp_multirag.cli`.

All hyperparameters of the four DP-MultiRAG modules are exposed as flags
so that the package can be driven from the command line without touching
Python code.
"""

from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dp_multirag",
        description=("DP-MultiRAG: per-document differential privacy for "
                     "multi-turn RAG."),
    )

    g = p.add_argument_group("LLM / API")
    g.add_argument("--base-url",
                   default=os.getenv("OPENAI_BASE_URL",
                                     "http://10.0.4.43:9999/v1"))
    g.add_argument("--api-key",
                   default=os.getenv("OPENAI_API_KEY", "sk-anything"))
    g.add_argument("--model",
                   default=os.getenv("OPENAI_MODEL", "qwen3-30b"))

    g = p.add_argument_group("Data")
    here = os.path.dirname(os.path.abspath(__file__))
    g.add_argument("--docs", default=os.path.join(here, "data", "docs.json"))
    g.add_argument("--turns", default=os.path.join(here, "data", "turns.json"))

    g = p.add_argument_group("Privacy budget (Module 4)")
    g.add_argument("--eps", type=float, default=1.0,
                   help="ε_global per document.")
    g.add_argument("--delta", type=float, default=1e-5,
                   help="δ used by the Gaussian mechanisms.")
    g.add_argument("--eps-ret-frac", type=float, default=0.30,
                   help="fraction of ε_global for retrieval.")
    g.add_argument("--eps-trans-frac", type=float, default=0.20,
                   help="fraction of ε_global for evidence transformation.")
    g.add_argument("--eps-gen-frac", type=float, default=0.50,
                   help="fraction of ε_global for generation.")
    g.add_argument("--decay", type=float, default=0.92,
                   help="cross-turn decay applied to cached exposure.")
    g.add_argument("--alpha1", type=float, default=1 / 3,
                   help="weight α_1 for n_i^ret in Exp(d_i).")
    g.add_argument("--alpha2", type=float, default=1 / 3,
                   help="weight α_2 for n_i^ctx in Exp(d_i).")
    g.add_argument("--alpha3", type=float, default=1 / 3,
                   help="weight α_3 for n_i^align in Exp(d_i).")

    g = p.add_argument_group("Module 1 — retrieval")
    g.add_argument("--top-k", type=int, default=10,
                   help="K in Gumbel Top-K.")
    g.add_argument("--gamma", type=float, default=0.4,
                   help="utility weight γ.")
    g.add_argument("--beta", type=float, default=0.6,
                   help="risk weight β.")
    g.add_argument("--w1", type=float, default=0.5,
                   help="weight on intrinsic sensitivity S(d_i).")
    g.add_argument("--w2", type=float, default=0.3,
                   help="weight on cumulative exposure Exp(d_i).")
    g.add_argument("--w3", type=float, default=0.2,
                   help="weight on alignment with the query.")

    g = p.add_argument_group("Module 2 — evidence transformation")
    g.add_argument("--phi-low", type=float, default=0.20,
                   help="Φ threshold below which evidence is left intact.")
    g.add_argument("--phi-high", type=float, default=0.55,
                   help="Φ threshold above which redaction is enforced.")
    g.add_argument("--tau-s", type=float, default=0.40,
                   help="sufficiency threshold τ_s in the three-way decision.")
    g.add_argument("--tau-b", type=float, default=0.05,
                   help="budget threshold τ_b in the three-way decision.")
    g.add_argument("--re-retrieve-factor", type=float, default=1.5,
                   help=("multiplier on top-K used by the retry pass when "
                         "the action is 'retrieve'."))

    g = p.add_argument_group("Reference-answer decode (Module 3)")
    g.add_argument("--reference-decode", action="store_true",
                   help=("Use pre-authored per-turn responses; Module 3 "
                         "charges Δε^gen ∝ disclosure_t · ε_t^gen instead "
                         "of running the LLM."))
    g.add_argument("--reference-answers",
                   default=os.path.join(here, "data", "reference_answers.json"),
                   help="JSON file with per-turn {answer, disclosure} entries.")

    g = p.add_argument_group("Misc")
    g.add_argument("--seed", type=int, default=20240601,
                   help="seed for the Gumbel + Gaussian samplers.")

    return p
