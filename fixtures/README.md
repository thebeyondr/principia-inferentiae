# Synthetic fixtures

All fixture values are synthetic unless a file explicitly carries measured provenance.

- Model, quantization, MHA/GQA/MQA, hardware, topology, and reserve fixtures live in
  `data/catalogs/` and typed pytest factories.
- Workload/SLO and planned-measurement fixtures live in
  `case-studies/long-context-bursty/inputs/`.
- Incident tests construct correlation-only, direct-OOM causal-timeline, and conflicting
  evidence contracts in `tests/test_sizing_and_diagnosis.py`.

Synthetic ceilings test arithmetic and evidence gates. They are never eligible for a
vendor, cost, production-sizing, or real-performance claim.

