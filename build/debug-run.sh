#!/usr/bin/env bash
set -u
B=/var/tmp/insignia-build/glm53-generate
gdb -batch -ex run -ex bt -ex "info locals" \
    --args $B /var/lib/insignia/glm53-flash-text /var/lib/insignia/glm53-flash-text.index \
    154820,13,171,1496,2343 0 12 /var/lib/insignia/glm53-fp8-g64 2>&1 | tail -40
