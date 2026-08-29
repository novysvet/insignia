python3 - <<'PYEOF'
import numpy as np
P = [10, 11, 12]
A = set([11, 12, 13])
cum = np.zeros(4)
for n in (1, 2, 3):
    v = P[n - 1] in A
    cum[n] += v
    print("n=", n, "in?", v, "cum:", cum)
x = np.zeros(4)
x[1] += 10 in A
print("direct +=:", x)
PYEOF
