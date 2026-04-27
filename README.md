# DP-MultiRAG

History-aware **per-document differential privacy** for multi-turn
Retrieval-Augmented Generation. The package implements the four modules
of the DP-MultiRAG paper as plain Python functions and ships a
command-line entry point that runs them end-to-end on a small example
corpus.

```
            ┌────────────────────────────────────────────────────────┐
  q_t ───►  │ M1  Exposure-aware Retrieval Control                   │
            │     R + γU − βΦ  →  Gumbel TopK  →  charge ε^ret       │
            └─────────────────────────────┬──────────────────────────┘
                                          ▼
            ┌────────────────────────────────────────────────────────┐
            │ M2  Controlled Evidence Utilization                    │
            │     redact / abstract / Gaussian-noise → suff → action │
            └─────────────────────────────┬──────────────────────────┘
                                          ▼
            ┌────────────────────────────────────────────────────────┐
            │ M3  Privacy-aware Generation                           │
            │     logit + N(0, σ²)  →  per-token η  →  charge ε^gen  │
            └─────────────────────────────┬──────────────────────────┘
                                          ▼
            ┌────────────────────────────────────────────────────────┐
            │ M4  Document-level Privacy Ledger                      │
            │     Exp(d) = α₁n^ret + α₂n^ctx + α₃n^align             │
            │     ε^global ≥ Σ_t (ε^ret_t + ε^trans_t + ε^gen_t)     │
            └────────────────────────────────────────────────────────┘
```

## Install

```bash
pip install -e .
```

The only runtime dependency is `openai>=1.30.0`.

## Quick start

```bash
export OPENAI_BASE_URL=http://localhost:9999/v1
export OPENAI_API_KEY=sk-anything
export OPENAI_MODEL=qwen3-30b

dp-multirag --eps 1.0 --top-k 10
```

`dp-multirag` is the entry point declared in `pyproject.toml`; it is
equivalent to `python -m dp_multirag.cli`.

## Python API

```python
import random
from dp_multirag import LLMClient, ledger as L
from dp_multirag.pipeline import per_turn_budget, run_turn
from dp_multirag.cli import load_json

llm   = LLMClient(base_url="http://localhost:9999/v1",
                  api_key="sk-anything", model="qwen3-30b")
docs  = load_json("dp_multirag/data/docs.json")
turns = load_json("dp_multirag/data/turns.json")

state = L.init_ledger(docs, eps_global=1.0, delta_global=1e-5)
budgets = per_turn_budget(eps_global=1.0,
                          eps_ret_frac=0.30,
                          eps_trans_frac=0.20,
                          eps_gen_frac=0.50,
                          n_turns=len(turns))

cfg = {**budgets,
       "top_k": 10, "gamma": 0.4, "beta": 0.6,
       "w": (0.5, 0.3, 0.2),
       "phi_low": 0.20, "phi_high": 0.55,
       "tau_s": 0.40,   "tau_b": 0.05}

rng = random.Random(20240601)
history = []
for q in turns:
    if history:
        L.decay_exposure(state, decay=0.92)
    res = run_turn(llm=llm, raw_q=q, history=history,
                   docs=docs, ledger=state, cfg=cfg, rng=rng)
    print(res.action, "→", res.answer)
    history.append((q, res.answer))

print(L.full_dump(state))
```

## Modules

### Module 1 — `dp_multirag.retrieval`

| Function                    | Role                                              |
|-----------------------------|---------------------------------------------------|
| `rewrite_query`             | History-aware rewriting `R(q_t, H_t)`             |
| `intrinsic_sensitivity`     | `S(d_i)` from tier-weighted span density          |
| `alignment`                 | `Align(d_i, q̃_t, H_t)`                            |
| `phi`                       | `Φ(d_i) = w₁S + w₂Exp + w₃Align`                  |
| `utility`                   | `U(d_i; C_<t)`                                    |
| `score_documents`           | `Score = Rel + γU − βΦ`                           |
| `gumbel_topk`               | DP Top-K via the exponential mechanism            |
| `charge_retrieval_costs`    | `Δε_i^{ret,t} = ε_t^{ret} · ρ_i^{(t)}`            |
| `retrieve`                  | full Module-1 pipeline as a single call           |

### Module 2 — `dp_multirag.evidence`

| Function                    | Role                                              |
|-----------------------------|---------------------------------------------------|
| `gaussian_sigma`            | analytical Gaussian-mechanism scale               |
| `transform_evidence`        | redact / abstract / Gaussian-noise per Φ          |
| `sufficiency`               | `r_t^{suf}` ∈ `[0, 1]`                            |
| `decide_action`             | three-way decision `{generate, retrieve, abstain}`|

### Module 3 — `dp_multirag.generation`

| Function                    | Role                                              |
|-----------------------------|---------------------------------------------------|
| `gaussian_sigma`            | `σ_gen = Δ_gen · √(2 ln(1.25/δ)) / ε_gen`          |
| `token_eta`                 | per-token risk weight `η_{t,τ}`                   |
| `privacy_decode`            | logit-noised step-by-step decoding                |
| `reference_decode`          | decode with a pre-authored response and `disclosure ∈ [0, 1]` |

### Module 4 — `dp_multirag.ledger`

| Function                    | Role                                              |
|-----------------------------|---------------------------------------------------|
| `init_ledger`               | per-document budgets and zero counters            |
| `exposure`                  | `Exp(d_i) = α₁n^ret + α₂n^ctx + α₃n^align`        |
| `total_used`, `remaining`,<br>`global_remaining` | budget queries                                    |
| `charge_retrieval`,<br>`charge_transform`,<br>`charge_generation` | three-component privacy consumption  |
| `decay_exposure`            | soft cross-turn decay on cached exposure          |
| `snapshot`, `full_dump`     | logging helpers                                   |

## Repository layout

```
dp-multirag/
├── README.md
├── pyproject.toml
├── requirements.txt
└── dp_multirag/
    ├── __init__.py
    ├── ledger.py            # Module 4
    ├── retrieval.py         # Module 1
    ├── evidence.py          # Module 2
    ├── generation.py        # Module 3
    ├── llm.py               # OpenAI / vLLM wrapper
    ├── pipeline.py          # run_turn(...) glueing M1–M4
    ├── cli.py               # python -m dp_multirag.cli
    ├── config.py            # argparse definitions
    ├── data/
    │   ├── docs.json
    │   ├── turns.json
    │   └── reference_answers.json
    └── examples/
        └── trace.txt
```

## Citation

```bibtex
@article{dp_multirag_2025,
  title  = {DP-MultiRAG: History-Aware Differential Privacy for Multi-Turn RAG},
  author = {...},
  year   = {2025}
}
```

## License

MIT.
