#!/usr/bin/env bash
# Show thread states/wchan of the generate process (hang diagnosis).
P=$(pgrep -f glm53-generate | head -1)
echo "pid=$P"
grep -e State -e Threads "/proc/$P/status"
for T in $(ls "/proc/$P/task"); do
  W=$(cat "/proc/$P/task/$T/wchan" 2>/dev/null)
  S=$(awk '{print $3}' "/proc/$P/task/$T/stat")
  echo "$T state=$S wchan=$W"
done | sort | uniq -c | sort -rn | head -20
