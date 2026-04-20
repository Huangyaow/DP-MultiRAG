# DP-RAG minimal demo

End-to-end run of the 6-step **History-Aware DP-RAG** pipeline against a
hard-coded multi-turn cumulative-leakage attack trajectory. Every
intermediate artefact (rewritten query, retrieved docs, risk score,
transformed evidence, answer, ledger snapshot) is printed to stdout.

## Layout

```
demo/
├── demo.py             # main entry: orchestrates the 6 steps
├── func.py             # one function per step (rewrite / retrieve / risk /
│                       #  transform / decode / ledger update) + LLM wrapper
├── args.py             # argparse
├── data/
│   ├── docs.json       # toy sensitive knowledge base
│   ├── turns.json      # 6-turn attack trajectory
│   └── README.md
├── requirements.txt
└── README.md
```

## Setup

The demo talks to an OpenAI-compatible endpoint. A local **vLLM** server
serving `qwen3-8b` on port 8003 is the default.

```bash
pip install -r requirements.txt
```

Verify the endpoint is reachable:

```bash
curl http://localhost:8003/v1/models
```

## Run

```bash
cd dp_rag/demo
python demo.py --model qwen3-8b --eps 3.0
```

Useful flags:

| flag              | default                       | meaning                                          |
|-------------------|-------------------------------|--------------------------------------------------|
| `--model`         | `qwen3-8b`                    | vLLM served model name                           |
| `--base-url`      | `http://localhost:8003/v1`    | OpenAI-compatible endpoint                       |
| `--eps`           | `3.0`                         | per-document privacy budget                      |
| `--risk-high`     | `0.65`                        | risk threshold for triggering evidence transform |
| `--budget-floor`  | `0.3`                         | refuse when remaining budget falls below this    |
| `--decay`         | `0.85`                        | exposure decay factor between turns              |
| `--top-k`         | `2`                           | candidates retrieved per turn                    |

## What you should see

- Turns 1-3 ask low-sensitivity meta questions (reporter, branch, incident
  type). Risk stays under the threshold, evidence is **not** redacted, the
  model answers normally.
- Turn 4 (`Which records were affected?`) hits the high-sensitivity span
  `employee IDs 4821-4835`. Risk crosses `--risk-high`, **Step 3** redacts
  the IDs, and the answer becomes a controlled refusal.
- Turn 5-6 deplete the per-document budget; the gate in `demo.py` returns a
  hard refusal without even calling the LLM.
