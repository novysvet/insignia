#!/usr/bin/env bash
# Long-context latent A/B: FP8 latent (default) vs FP32 latent (KV_FP8=0),
# scalar decode, LOGITS_DUMP + compare_logits.py + ppl.py.
# Usage: longctx-ab.sh [prompt_tokens] [gen_tokens]
set -uo pipefail
G=/var/tmp/insignia-build/glm53-generate
M=/var/lib/insignia/glm53-flash-text
I=/var/lib/insignia/glm53-flash-text.index
Q=/var/lib/insignia/glm53-fp8-g64
OUT=/var/lib/insignia/longctx
REPO=/mnt/c/coding/Insignia-glm53-dflash2
PTOK="${1:-500}"
GEN="${2:-16}"
mkdir -p "$OUT"

/var/lib/insignia/bench-venv/bin/python - "$PTOK" > "$OUT/ids.csv" <<'EOF'
import sys
import pyarrow.parquet as pq
from tokenizers import Tokenizer
n = int(sys.argv[1])
tok = Tokenizer.from_file("/var/lib/insignia/glm53-flash-text/tokenizer.json")
rows = pq.read_table("/var/lib/insignia/bench-data/gsm8k/main/test-00000-of-00001.parquet").to_pylist()
text = "\n".join(r["question"] for r in rows[:40])
ids = tok.encode(text).ids[:n]
print(",".join(map(str, ids)))
EOF
echo "prompt tokens: $(tr ',' '\n' < "$OUT/ids.csv" | wc -l)"

run() { # name kvfp8
  local name="$1" kv="$2"
  INSIGNIA_GLM53_Q8_BUDGET_MB=10240 INSIGNIA_GLM53_EXPERT_CACHE_MB=32768 \
  INSIGNIA_GLM53_KV_FP8="$kv" INSIGNIA_GLM53_LOGITS_DUMP="$OUT/$name.f32" \
  $G "$M" "$I" "$(cat "$OUT/ids.csv")" 0 "$GEN" "$Q" > "$OUT/$name.log" 2>&1
  grep -E "greedy IDs|tokens total" "$OUT/$name.log" | sed "s/^/[$name] /"
}

run a8 1
run b32 0
echo "== compare (FP8-latent vs FP32-latent) =="
python3 "$REPO/tools/compare_logits.py" "$OUT/a8.f32" "$OUT/b32.f32" --tokens "$OUT/ids.csv" || true
