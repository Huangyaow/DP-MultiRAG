# Demo data

- `docs.json`: small sensitive knowledge base. Each document has a list of
  `sensitive_spans` used by the risk assessor and the evidence transformer.
- `turns.json`: a 6-turn cumulative-leakage attack trajectory. The attacker
  asks low-sensitivity meta questions in the first 3 turns, then targets the
  high-sensitivity fields (employee IDs, mitigation details, schedule) in
  turns 4-6.
