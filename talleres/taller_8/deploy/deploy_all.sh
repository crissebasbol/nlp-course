#!/usr/bin/env bash
# Full deploy: check machines, LLM on Machine 1, app on Machine 2.
set -euo pipefail
DIR="$(dirname "$0")"
"$DIR/check_servers.sh"
"$DIR/deploy_llm.sh"
"$DIR/deploy_app.sh"
"$DIR/status.sh"
