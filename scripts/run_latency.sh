#!/usr/bin/env bash
# Time every model back to back in one uninterrupted sequence.
#
# Latency is only comparable across models when they are measured together: this
# shared 4-core box drifts ~25% between batches, enough to invent a difference
# between two models that are byte-identical. Accuracy runs can be split across
# batches (they are deterministic); latency cannot.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "== back-to-back latency pass =="
uv run --project fastsam         python scripts/latency_pass.py fastsam x        2>&1 | tail -1
uv run --project fastsam         python scripts/latency_pass.py fastsam s        2>&1 | tail -1
uv run --project mobilesam       python scripts/latency_pass.py mobilesam        2>&1 | tail -1
uv run --project edgesam         python scripts/latency_pass.py edgesam          2>&1 | tail -1
uv run --project tinysam         python scripts/latency_pass.py tinysam          2>&1 | tail -1
uv run --project repvitsam       python scripts/latency_pass.py repvitsam        2>&1 | tail -1
uv run --project efficientsam    python scripts/latency_pass.py efficientsam vitt 2>&1 | tail -1
uv run --project efficientsam    python scripts/latency_pass.py efficientsam vits 2>&1 | tail -1
uv run --project efficientvitsam python scripts/latency_pass.py efficientvitsam l0 2>&1 | tail -1
uv run --project efficientsam3   python scripts/latency_pass.py efficientsam3 tinyvit 2>&1 | tail -1
echo "== done =="
