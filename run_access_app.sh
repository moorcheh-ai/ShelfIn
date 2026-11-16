#!/usr/bin/env bash
# Back-compat wrapper — prefer: bash ./start.sh
exec bash "$(cd "$(dirname "$0")" && pwd)/start.sh" "$@"
