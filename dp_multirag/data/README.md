# Data

Three small JSON files used by `dp_multirag.cli` and reusable from the
public Python API.

## `docs.json`

The sensitive knowledge base. Each document is

```json
{
  "id": "kb_002_alice_payroll",
  "text": "Alice (employee ID 47281) reported a payroll-related data leak ...",
  "sensitive_spans": [
    {"text": "Alice",       "tier": "low"},
    {"text": "47281",       "tier": "high"},
    ...
  ]
}
```

Tiers drive Module 2 (`evidence.transform_evidence`):

| Tier   | Operator                                |
|--------|-----------------------------------------|
| high   | redaction (or abstraction at low Φ)     |
| low    | abstraction                             |

Numeric tokens are perturbed with calibrated Gaussian noise.

## `turns.json`

The list of raw user queries that compose a multi-turn trajectory.

```json
[
  "Did anyone in our company report the incident?",
  ...
]
```

## `reference_answers.json`

Pre-authored per-turn responses used when running the CLI with
`--reference-decode`. Each entry is

```json
{
  "answer":     "Alice reported the incident.",
  "disclosure": 1.0,
  "comment":    "...optional human-readable label..."
}
```

`disclosure ∈ [0, 1]` is the per-turn fraction of `ε_t^gen` that this
response is treated as having consumed. A hard refusal at `disclosure = 0`
consumes zero ε^gen; a full disclosure at `disclosure = 1` consumes the
full per-turn generation budget.
