#!/usr/bin/env bash
set -eo pipefail

# Ascend's set_env.sh appends to these variables, while nohup may start
# without them. Initialize them before entering the strict queue script.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"

exec bash /data1/ma00959358/pangu/ToolRGSNPU/tools/run_three_baselines_bs256.sh
