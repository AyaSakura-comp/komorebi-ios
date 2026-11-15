#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8765}"
MODELS_DIR="${HOME}/models"

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "Models directory not found: $MODELS_DIR" >&2
  exit 1
fi

IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")
echo "Serving Komorebi offline models from: $MODELS_DIR"
echo "Enter this URL inside iOS App -> 模型與效能 -> LAN Companion Server:"
echo ""
echo "    http://${IP}:${PORT}"
echo ""
echo "Press Ctrl+C to stop."
exec python3 -m http.server "$PORT" --directory "$MODELS_DIR"
