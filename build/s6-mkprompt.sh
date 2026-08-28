#!/usr/bin/env bash
# Emit a GSM8K-derived prompt CSV of N tokens to stdout. Usage: s6-mkprompt.sh N [ROW_OFFSET]
set -euo pipefail
/var/lib/insignia/bench-venv/bin/python - "$1" "${2:-0}" <<'EOF'
import sys
import pyarrow.parquet as pq
from tokenizers import Tokenizer
n = int(sys.argv[1])
offset = int(sys.argv[2])
tok = Tokenizer.from_file("/var/lib/insignia/glm53-flash-text/tokenizer.json")
rows = pq.read_table("/var/lib/insignia/bench-data/gsm8k/main/test-00000-of-00001.parquet").to_pylist()
text = "\n".join(r["question"] for r in rows[offset:offset + 40])
ids = tok.encode(text).ids[:n]
print(",".join(map(str, ids)))
EOF
