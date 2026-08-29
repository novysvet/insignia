#!/usr/bin/env bash
# Merge per-run ROUTE_TRACE files into one token-renumbered trace + manifest.
#
# Usage: merge_traces.sh [CAMPAIGN_DIR] [OUTPUT_PREFIX]
#   CAMPAIGN_DIR   default /var/lib/insignia/tracecampaign
#                  (reads traces/route-<id>.tt per prompts/manifest.tsv; legacy
#                   /var/lib/insignia/route-{realtext,campaign}.txt fold in first)
#   OUTPUT_PREFIX  default <CAMPAIGN_DIR>/merged/route-merged
#                  -> $PREFIX.trace (rows) + $PREFIX.manifest.tsv (run table)
#
# Format v1 (see TRACE-FORMAT.md): rows `<token> <layer> e0..e7 s0..s7`
# (18 fields), token-major, the 42 sparse-layer rows of one decode token
# contiguous, token ids globally unique: run #k gets token base k*100000, so
# ids encode provenance and dump_cct.py's by-token grouping stays correct.
# make_pinlist.py and dump_cct.py consume $PREFIX.trace unchanged.
set -uo pipefail

DIR="${1:-/var/lib/insignia/tracecampaign}"
PREFIX="${2:-$DIR/merged/route-merged}"
mkdir -p "$(dirname "$PREFIX")"
: > "$PREFIX.trace"
printf 'run\tfile\trows\ttokens\ttoken_base\tdataset\trow\n' > "$PREFIX.manifest.tsv"

K=0
emit () {  # emit <id> <file> [dataset row]
    local id="$1" file="$2" ds="${3:--}" row="${4:--}" base rows tokens
    [ -f "$file" ] || return 0
    base=$((K * 100000))
    rows=$(awk -v base="$base" 'NF >= 11 {
        printf "%d %s", $1 + base, $2
        for (i = 3; i <= NF; i++) printf " %s", $i
        printf "\n" }' "$file" >> "$PREFIX.trace"; wc -l < "$file")
    tokens=$(awk 'NF >= 11 { s[$1] = 1 } END { print length(s) }' "$file")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$file" "$rows" "$tokens" "$base" "$ds" "$row" \
        >> "$PREFIX.manifest.tsv"
    echo "  $id: $tokens tokens ($rows rows, base $base)" >&2
    K=$((K + 1))
}

for legacy in /var/lib/insignia/route-realtext.txt /var/lib/insignia/route-campaign.txt; do
    [ -f "$legacy" ] && emit "legacy-$(basename "$legacy" .txt)" "$legacy"
done
if [ -f "$DIR/prompts/manifest.tsv" ]; then
    while IFS=$'\t' read -r id dataset row ptok gen; do
        [ "$id" = "id" ] && continue
        emit "$id" "$DIR/traces/route-$id.txt" "$dataset" "$row"
    done < "$DIR/prompts/manifest.tsv"
fi

echo "merged -> $PREFIX.trace ($(wc -l < "$PREFIX.trace") rows)"
echo "split protocol: drop legacy + p00 rows, alternate runs -> train/test (TRACE-FORMAT.md)"
