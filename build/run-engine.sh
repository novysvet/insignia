#!/usr/bin/env bash
# Direct engine run with visible errors. Usage: run-engine.sh "TOKENS" GENERATE [EXTRA_ENV]
set -u
B=/var/tmp/insignia-build/glm53-generate
ROOT=/var/lib/insignia/glm53-flash-text
IDX=/var/lib/insignia/glm53-flash-text.index
Q8=/var/lib/insignia/glm53-fp8-g64
TOKS="${1:-154820,13,171,1496,2343}"
GEN="${2:-12}"
if [ $# -ge 3 ]; then
  env "$3" $B $ROOT $IDX "$TOKS" 0 "$GEN" $Q8
else
  $B $ROOT $IDX "$TOKS" 0 "$GEN" $Q8
fi
