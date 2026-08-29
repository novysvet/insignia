#!/usr/bin/env bash
# Wave A pipeline: packed-sidecar parity gate + packed A/B bench + route-trace
# campaign. Runs on glm-box inside WSL Arch, serialized GPU use throughout.
# Logs to /var/lib/insignia/wave-a/run.log; touches DONE at the end.
set -uo pipefail
OUT=/var/lib/insignia/wave-a
BIN=/var/tmp/insignia-build-raptor/glm53-generate
MODEL=/var/lib/insignia/glm53-flash-text
INDEX=/var/lib/insignia/glm53-flash-text.index
FP8=/var/lib/insignia/glm53-fp8-g64
DFP8=/var/lib/insignia/glm53-dflash2-fp8-fixed
PACKED=/var/lib/insignia/glm53-experts-nvfp4x.igx
PY=/var/lib/insignia/bench-venv/bin/python
REPO=/mnt/c/coding/Insignia-glm53-dflash2
mkdir -p "$OUT"
exec >>"$OUT/run.log" 2>&1
echo "=== wave-a start $(date -Is) ==="

if pgrep -af 'glm53-(generate|stream-bench|ops-bench|q8-bench|expert-bench)' >/dev/null; then
    echo "guard: engine already running; aborting" >&2
    echo "=== wave-a DONE (aborted: engine busy) $(date -Is) ==="
    touch "$OUT/DONE"
    exit 1
fi

base_env() {
    export INSIGNIA_GLM53_Q8_BUDGET_MB=10240
    export INSIGNIA_GLM53_EXPERT_CACHE_MB=32768
    export INSIGNIA_GLM53_READERS=4
    unset INSIGNIA_GLM53_DFLASH2 INSIGNIA_GLM53_DFLASH2_FP8 INSIGNIA_GLM53_DF_VERIFY_K \
          INSIGNIA_GLM53_DF_ADAPTIVE_K INSIGNIA_GLM53_PACKED_EXPERTS
}

P1="154820,13,171,1496,2343"
P2="154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25"

run() { # run <name> <prompt> <gen> [packed]
    base_env
    [ "${4:-}" = packed ] && export INSIGNIA_GLM53_PACKED_EXPERTS="$PACKED"
    "$BIN" "$MODEL" "$INDEX" "$2" 0 "$3" "$FP8" < /dev/null > "$OUT/$1.log" 2>&1
    echo "[$1] exit=$?"
    grep -E '^greedy IDs' "$OUT/$1.log" > "$OUT/$1.ids" || true
    grep -E 'ms/token|NVFP4 cache|VRAM expert tier|packed experts:|packed expand' "$OUT/$1.log" \
        | head -6 | sed "s/^/[$1] /"
}

df_run() { # df_run <name> <prompt> <gen> [packed] [adaptive]
    base_env
    export INSIGNIA_GLM53_DFLASH2=1 INSIGNIA_GLM53_DFLASH2_FP8="$DFP8" \
           INSIGNIA_GLM53_DF_VERIFY_K=7
    [ "${5:-}" = adaptive ] || export INSIGNIA_GLM53_DF_ADAPTIVE_K=0
    [ "${4:-}" = packed ] && export INSIGNIA_GLM53_PACKED_EXPERTS="$PACKED"
    "$BIN" "$MODEL" "$INDEX" "$2" 0 "$3" "$FP8" < /dev/null > "$OUT/$1.log" 2>&1
    echo "[$1] exit=$?"
    grep -E '^greedy IDs' "$OUT/$1.log" > "$OUT/$1.ids" || true
    grep -E 'ms/token|accepted histogram|NVFP4 cache|packed experts:|packed expand' "$OUT/$1.log" \
        | head -6 | sed "s/^/[$1] /"
}

echo "--- parity: scalar reference ---"
run a1-scalar "$P1" 40
run a2-scalar "$P2" 100
echo "--- parity: scalar packed (codec gate) ---"
run b1-scalar-packed "$P1" 40 packed
run b2-scalar-packed "$P2" 100 packed
echo "--- parity: dflash k7 fixed (driver-change gate) ---"
df_run c2-dflash "$P2" 100
echo "--- parity: dflash k7 fixed packed ---"
df_run d2-dflash-packed "$P2" 100 packed
echo "--- parity: dflash k7 adaptive (EMA change smoke) ---"
df_run e2-dflash-adaptive "$P2" 100 "" adaptive

echo "--- parity diffs ---"
fail=0
for pair in "a1-scalar b1-scalar-packed" "a2-scalar b2-scalar-packed" \
            "a2-scalar c2-dflash" "a2-scalar d2-dflash-packed" "a2-scalar e2-dflash-adaptive"; do
    set -- $pair
    if diff -q "$OUT/$1.ids" "$OUT/$2.ids" >/dev/null 2>&1 && [ -s "$OUT/$1.ids" ]; then
        echo "PARITY OK: $1 == $2"
    else
        echo "PARITY FAIL: $1 != $2"
        diff "$OUT/$1.ids" "$OUT/$2.ids" 2>/dev/null | head -4
        fail=1
    fi
done

echo "--- quick A/B bench: packed off vs on ---"
cd "$REPO" || fail=1
if [ "$fail" -eq 0 ]; then
    base_env
    "$PY" tools/benchmark_math.py --binary "$BIN" --output "$OUT/bench-unpacked" \
        --dflash-fp8 "$DFP8" --samples 2 --generate 32 --verify-k 7 \
        --q8-budget-mb 10240 --cache-mb 32768 --readers 4 || echo "bench unpacked FAILED"
    INSIGNIA_GLM53_PACKED_EXPERTS="$PACKED" \
    "$PY" tools/benchmark_math.py --binary "$BIN" --output "$OUT/bench-packed" \
        --dflash-fp8 "$DFP8" --samples 2 --generate 32 --verify-k 7 \
        --q8-budget-mb 10240 --cache-mb 32768 --readers 4 || echo "bench packed FAILED"
    echo "--- bench-unpacked summary ---"; tail -12 "$OUT/bench-unpacked/summary.md" 2>/dev/null
    echo "--- bench-packed summary ---"; tail -12 "$OUT/bench-packed/summary.md" 2>/dev/null
fi
echo "=== wave-a bench complete $(date -Is) parity_fail=$fail ==="

if [ "$fail" -eq 0 ]; then
    echo "--- route-trace campaign (~3h) ---"
    bash /var/lib/insignia/tracecampaign/campaign.sh || echo "campaign exited nonzero"
    bash /var/lib/insignia/tracecampaign/merge_traces.sh || true
    echo "=== campaign complete $(date -Is) ==="
else
    echo "!!! parity failed; skipping campaign"
fi
echo "=== wave-a DONE $(date -Is) parity_fail=$fail ==="
touch "$OUT/DONE"
