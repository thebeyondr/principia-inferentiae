.PHONY: verify format case schemas

verify:
	ruff check .
	ruff format --check .
	mypy src tests
	pytest

format:
	ruff check --fix .
	ruff format .

case:
	principia prepare-vllm-trace --trace case-studies/long-context-bursty/inputs/trace.jsonl --chunk-hash-size 512 --output case-studies/long-context-bursty/outputs/vllm-timed-trace.jsonl
	principia analyze --run case-studies/long-context-bursty/inputs/run.yaml --trace case-studies/long-context-bursty/inputs/trace.jsonl --slo case-studies/long-context-bursty/inputs/slo.yaml --model case-studies/long-context-bursty/inputs/model.yaml --hardware case-studies/long-context-bursty/inputs/hardware.yaml --config case-studies/long-context-bursty/inputs/serving-config.yaml --measurement case-studies/long-context-bursty/inputs/roofline-measurement.yaml --quality-gate case-studies/long-context-bursty/inputs/quality-gate.yaml --policy case-studies/long-context-bursty/inputs/sizing-policy.yaml --output case-studies/long-context-bursty/outputs/analysis.json
	principia write-brief --analysis case-studies/long-context-bursty/outputs/analysis.json --audience engineering --output case-studies/long-context-bursty/report.md
	principia write-brief --analysis case-studies/long-context-bursty/outputs/analysis.json --audience stakeholder --output case-studies/long-context-bursty/outputs/stakeholder-brief.md

schemas:
	principia export-schemas --output-directory schemas

