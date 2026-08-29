#!/usr/bin/env bash
# Dev-box (E:\coding\Insignia, Git Bash) deploy helper for the bench matrix.
# Run BY THE OPERATOR (never automatically): it pushes code, pulls + builds
# on glm-box, stages the harness, and registers the Windows Task Scheduler
# entry that survives WSL VM recycles.
#
#   bash scratch/bench/deploy-matrix.sh [--stage singles|combos|all] \
#        [--time HH:MM] [--run-now] [--no-push] [--no-build]
#
# Defaults: --stage singles --time 23:30, task name InsigniaBenchMatrix.
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH=glm53-dflash2-4070ti-super
WORKTREE='C:\coding\Insignia-glm53-dflash2'          # NEVER C:\coding\Insignia (stale snapshot)
REMOTE_DIR=/mnt/c/coding/Insignia-glm53-dflash2
STAGE=singles
TIME=23:30
TASK=InsigniaBenchMatrix
PUSH=1
BUILD=1
RUNNOW=0

while [ $# -gt 0 ]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --time)  TIME="$2";  shift 2 ;;
        --task)  TASK="$2";  shift 2 ;;
        --run-now) RUNNOW=1; shift ;;
        --no-push) PUSH=0;   shift ;;
        --no-build) BUILD=0; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

[ "$(git -C "$REPO" branch --show-current)" = "$BRANCH" ] \
    || { echo "not on $BRANCH (on $(git -C "$REPO" branch --show-current))"; exit 2; }

echo "== 1. code: push $BRANCH (skips if nothing to push) =="
if [ "$PUSH" = 1 ]; then
    if [ -n "$(git -C "$REPO" log "origin/$BRANCH..HEAD" --oneline 2>/dev/null)" ]; then
        git -C "$REPO" push "origin $BRANCH"
    else
        echo "no unpushed commits; skipping push"
    fi
fi

echo "== 2. harness: stage scratch/bench -> glm-box worktree (scratch/ is untracked; tar over ssh) =="
ssh glm-box "wsl -d Arch -- bash -c 'mkdir -p $REMOTE_DIR/scratch/bench'"
tar -C "$REPO/scratch/bench" -czf - bench-matrix.sh bench-matrix-inner.sh \
    bench-matrix-task.cmd deploy-matrix.sh summarize-matrix.py MATRIX.md \
    | ssh glm-box "wsl -d Arch -- bash -c 'tar -xzf - -C $REMOTE_DIR/scratch/bench && sed -i \"s/\\r$//\" $REMOTE_DIR/scratch/bench/*.sh'"

echo "== 3. code: pull + build inside Arch WSL (raptor-tuned) =="
ssh glm-box "git -C $WORKTREE pull --ff-only"
if [ "$BUILD" = 1 ]; then
    ssh glm-box "wsl -d Arch -- bash -c 'cd $REMOTE_DIR && INSIGNIA_BUILD_DIR=/var/tmp/insignia-build-raptor bash build/glm53-gen.sh'"
else
    echo "build skipped (--no-build); binary must already exist at /var/tmp/insignia-build-raptor/glm53-generate"
fi

echo "== 4. args file + pre-flight inventory =="
ssh glm-box "wsl -d Arch -- bash -c 'echo $STAGE > /var/lib/insignia/bench-matrix-args'"
ssh glm-box "wsl -d Arch -- bash -c '
    pgrep -af \"glm53-|pack_glm53_experts|benchmark_math\" && { echo GPU BUSY - deploy stopped before scheduling; exit 1; }
    ls -l /var/tmp/insignia-build-raptor/glm53-generate
    ls -l /var/lib/insignia/glm53-experts-nvfp4x.igx 2>/dev/null || echo \"packed sidecar MISSING (packed-on + combos will SKIP)\"
    ls -l /var/lib/insignia/pinlist-v1.txt 2>/dev/null || echo \"pin list v1 MISSING (pin-v1 will SKIP)\"
    ls -l /var/lib/insignia/tracecampaign/pinlist-v2.txt 2>/dev/null || echo \"pin list v2 MISSING (pin-v2 + combos will SKIP; build it: /var/lib/insignia/bench-venv/bin/python tools/make_pinlist.py /var/lib/insignia/tracecampaign/route-merged.trace /var/lib/insignia/tracecampaign/pinlist-v2.txt)\"
    true'"

echo "== 5. register scheduled task ($TASK at $TIME) =="
ssh glm-box "schtasks /Create /F /SC ONCE /ST $TIME /TN $TASK /TR \"C:\\coding\\Insignia-glm53-dflash2\\scratch\\bench\\bench-matrix-task.cmd\""
[ "$RUNNOW" = 1 ] && ssh glm-box "schtasks /Run /TN $TASK"

cat <<EOF

Deployed. Monitoring from the dev box:
  ssh glm-box "wsl -d Arch -- tail -20 /var/lib/insignia/bench-matrix-task.log"
  ssh glm-box "wsl -d Arch -- cat /var/lib/insignia/bench-results/*-matrix/progress.tsv"
  ssh glm-box "wsl -d Arch -- bash $REMOTE_DIR/scratch/bench/bench-matrix.sh summarize"
Resume after any WSL recycle / crash: just re-run the task (done cells skip):
  ssh glm-box "schtasks /Run /TN $TASK"
Combos after the singles review:
  bash scratch/bench/deploy-matrix.sh --stage combos --run-now --no-push --no-build
EOF
