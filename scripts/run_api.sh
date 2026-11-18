#!/usr/bin/env bash
# Laptop 1: start the PathSnap API.  Usage:  bash scripts/run_api.sh [partition]   (default: eval)
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PATHSNAP_PARTITION="${1:-${PATHSNAP_PARTITION:-eval}}"
IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
echo "Partition: $PATHSNAP_PARTITION"
echo "Laptop 2 must use:  http://${IP:-<this-laptop-ip>}:8000"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
