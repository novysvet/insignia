#!/usr/bin/env bash
# Session 8 box health check — runs INSIDE Arch on glm-box (piped via ssh+wsl stdin).
echo "=== uptime ==="; uptime
echo "=== engine processes ==="; pgrep -af glm53-generate || echo NONE
echo "=== other GPU processes ==="; pgrep -af 'python|bench' | head -20 || echo NONE
echo "=== RAM/pinned pressure ==="; free -g
echo "=== build dirs ==="; ls -d /var/tmp/insignia-build* 2>/dev/null; ls -la /var/tmp/insignia-build-raptor/glm53-generate 2>/dev/null
echo "=== trace campaign state ==="; ls /var/lib/insignia/tracecampaign/ 2>/dev/null | wc -l; ls /var/lib/insignia/tracecampaign/ 2>/dev/null | tail -5
echo "=== bench-data ==="; ls /var/lib/insignia/bench-data/ 2>/dev/null
echo "=== bench results ==="; ls /var/lib/insignia/bench-results/ 2>/dev/null
echo "=== stores ==="; du -sh /var/lib/insignia/glm53-flash-text* /var/lib/insignia/glm53-fp8-g64 /var/lib/insignia/glm53-dflash2-fp8-fixed /var/lib/insignia/glm53-experts-packed* 2>/dev/null
echo "=== wave-a task ==="; ls /var/lib/insignia/wave-a/ 2>/dev/null | tail -5; ls /mnt/c/coding/ 2>/dev/null
echo "=== nvme ==="; df -h / /tmp | tail -2
