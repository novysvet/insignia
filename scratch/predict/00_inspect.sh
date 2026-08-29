set -e
cd /var/lib/insignia
echo "=== trace files ==="
ls -la route-realtext.txt route-campaign.txt early-route-math.txt cct-gsm8k.table 2>&1
echo "=== line counts ==="
wc -l route-realtext.txt route-campaign.txt early-route-math.txt cct-gsm8k.table 2>/dev/null
echo "=== heads ==="
for f in route-realtext.txt route-campaign.txt early-route-math.txt cct-gsm8k.table; do
  [ -f "$f" ] && { echo "--- $f ---"; head -3 "$f"; }
done
echo "=== dfdump ==="
ls -la dfdump/ | head -40
echo "=== dfdump file count / total size ==="
find dfdump -type f | wc -l
du -sh dfdump
echo "=== sample of a few dfdump files ==="
for f in $(find dfdump -type f | head -5); do echo "--- $f ---"; head -c 600 "$f" | cat -v | head -10; echo; done
