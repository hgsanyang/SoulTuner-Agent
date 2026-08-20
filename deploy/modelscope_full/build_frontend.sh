#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}/web"

npm ci
# Rewrites are compiled into the Next.js artifact.  All three processes share
# one container in this candidate, so the private loopback address is stable.
NEXT_PUBLIC_API_URL="" BACKEND_INTERNAL_URL="http://127.0.0.1:8501" npm run build
test -f .next/standalone/server.js
