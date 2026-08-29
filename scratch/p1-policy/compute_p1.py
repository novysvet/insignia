# P1 verification arithmetic: union extrapolation, cost model, k* tables,
# counterexample, break-even surfaces, sequential-vs-batch, heuristic loss.
# Pure float math, no dependencies. Run: py compute_p1.py

import math

S = 570.9          # ms/token, measured scalar decode median
KMAX = 8

def wavg(pairs):
    tot = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / tot

# ---------------- 1. Union curve U(k) per layer, k=1..8 ----------------
U = {1: 8.0, 2: 14.45, 3: 20.61, 4: 26.40, 5: 31.40}
incs = [U[k] - U[k-1] for k in (2, 3, 4, 5)]           # 6.45 6.16 5.79 5.00
r_inc = (incs[-1] / incs[0]) ** (1/3)                   # geometric decay of increments
UA = dict(U); dd = incs[-1]
for k in range(6, 9):
    dd *= r_inc
    UA[k] = UA[k-1] + dd

def neff(k):  # effective independent draws matching measured U(k)
    return -288.0 * math.log(1.0 - U[k] / 288.0)
phi = {k: neff(k) / (8*k) for k in range(2, 6)}
r_phi = (phi[5] / phi[2]) ** (1/3)                      # decay of clumping factor
UB = dict(U); p = phi[5]
for k in range(6, 9):
    p *= r_phi
    UB[k] = 288.0 * (1.0 - math.exp(-8.0 * k * p / 288.0))

Um = {k: (UA[k] + UB[k]) / 2 for k in range(1, 9)}
ratio = {k: Um[k] / (8*k) for k in range(1, 9)}
d = {k: 42.0 * Um[k] for k in range(1, 9)}              # expected records/round
ddrec = {k: d[k] - d[k-1] for k in range(2, 9)}

print("=== union curve ===")
print(f"r_inc={r_inc:.4f}  r_phi={r_phi:.4f}")
for k in range(1, 9):
    print(f"k={k}: UA={UA[k]:6.2f} UB={UB[k]:6.2f} U={Um[k]:6.2f} ratio={ratio[k]:.4f} "
          f"d={d[k]:7.1f} Dd={ddrec.get(k,0):6.1f}  b*Dd(0.6)={0.6*ddrec.get(k,336-336+336*0):6.1f}" if k>1 else
          f"k={k}: UA={UA[k]:6.2f} UB={UB[k]:6.2f} U={Um[k]:6.2f} ratio={ratio[k]:.4f} d={d[k]:7.1f}")

def C(k, a, b):  # ms per batch verify round
    return a + b * d[k]

def L_of(q, k):  # q[j] = P(first j drafts all accepted), j=1..7
    return 1.0 + sum(q[j-1] for j in range(1, k))

def T_table(q, a, b, f=0.0):
    out = []
    for k in range(1, KMAX + 1):
        c = C(k, a, b) + f * (1.0 - q[0])
        out.append(L_of(q, k) / c * 1000.0)  # tok/s
    return out

def argmax_k(ts, kmin=1):
    best = kmin
    for k in range(kmin, len(ts) + 1):
        if ts[k-1] > ts[best-1] + 1e-12:
            best = k
    return best

def marginal_rule(q, a, b):
    # start k=1, advance while DeltaL/DeltaC > T(k)
    k = 1
    Lk, Ck = L_of(q, 1), C(1, a, b)
    while k < KMAX:
        dL = q[k-1]
        dC = C(k+1, a, b) - C(k, a, b)
        if dL / dC > Lk / Ck:
            k += 1
            Lk, Ck = L_of(q, k), C(k, a, b)
        else:
            break
    return k

# ---------------- regimes (q vectors, q_j = P(M >= j)) ----------------
def qPAR(w, beta):  return [w + (1-w) * beta**j for j in range(1, 8)]
def qAG(pi0, rho):  return [(1-pi0) * rho**j for j in range(1, 8)]

regimes = {
    "campaign-meas (L4=3.70,L7=5.88)": [0.98, 0.90, 0.82, 0.74, 0.73, 0.72, 0.72],
    "PAR(0.76,0.6)": qPAR(0.76, 0.6),
    "PAR(0.90,0.5)": qPAR(0.90, 0.5),
    "AG(0.15,0.735) medium": qAG(0.15, 0.735),
    "real-A (L4=2.64,L7=2.91)": [0.64, 0.55, 0.45, 0.12, 0.08, 0.07, 0.06],
    "real-B (L7=4.57)": [0.70, 0.66, 0.62, 0.60, 0.55, 0.44, 0.40],
    "L2 parrot-broken (L4=1.58)": qAG(0.474, 0.578),
}

print("\n=== regime fits, a=100, b=0.6 ===")
for name, q in regimes.items():
    Ls = [L_of(q, k) for k in range(1, 9)]
    ts = T_table(q, 100, 0.6)
    ks = argmax_k(ts, 2)
    mr = marginal_rule(q, 100, 0.6)
    print(f"{name:32s} q={[round(x,3) for x in q]}")
    print(f"   L={ [round(x,3) for x in Ls] }")
    print(f"   T={ [round(x,3) for x in ts] } tok/s   k*={ks} T*={ts[ks-1]:.3f} ({1000/ts[ks-1]:.1f} ms/tok)  marginal_rule->{mr}"
          f"  recs/tok={d[ks]/Ls[ks-1]:.0f}")

print("\n=== regime k* sensitivity over (a,b) ===")
for a in (17, 60, 100, 150):
    for b in (0.6, 1.22):
        row = []
        for name, q in regimes.items():
            ts = T_table(q, a, b)
            row.append(f"{argmax_k(ts,2)}")
        print(f"a={a:3d} b={b:.2f}: k* per regime = {row}")

# ---------------- counterexample ----------------
print("\n=== counterexample (a=100, b=0.6) ===")
qce = [1.00, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61]
ts = T_table(qce, 100, 0.6)
print("T =", [round(t, 4) for t in ts])
print("marginal rule stops at k =", marginal_rule(qce, 100, 0.6),
      " argmax =", argmax_k(ts, 2), f" loss = {(max(ts[1:])/ts[1]-1)*100:.1f}%")
ts60 = T_table(qce, 60, 0.6)
print("a=60: T =", [round(t, 4) for t in ts60], " rule->", marginal_rule(qce, 60, 0.6), " argmax->", argmax_k(ts60, 2))
# exchangeable near-miss
qex = [0.60, 0.53, 0.51, 0.49, 0.47, 0.45, 0.43]
tse = T_table(qex, 100, 0.6)
print("exchangeable variant T =", [round(t,4) for t in tse], " rule->", marginal_rule(qex, 100, 0.6), " argmax->", argmax_k(tse,2))

# ---------------- k* grids ----------------
print("\n=== PAR(w,beta) k* grid (a=100,b=0.6) ===")
print("        beta=" + "".join(f"{b:<14}" for b in (0.3, 0.5, 0.7)))
for w in (0.2, 0.4, 0.6, 0.76, 0.9):
    row = []
    for beta in (0.3, 0.5, 0.7):
        q = qPAR(w, beta); ts = T_table(q, 100, 0.6); ks = argmax_k(ts, 2)
        row.append(f"k*={ks} {ts[ks-1]:5.2f}t/s")
    print(f"w={w:4.2f}: " + "".join(f"{c:<16}" for c in row))

print("\n=== AG(pi0,rho) k* grid (a=100,b=0.6) ===")
for pi0 in (0.0, 0.15, 0.30, 0.474):
    row = []
    for rho in (0.4, 0.578, 0.7, 0.8, 0.9):
        q = qAG(pi0, rho); ts = T_table(q, 100, 0.6); ks = argmax_k(ts, 2)
        on = ts[ks-1] > 1000.0/S
        row.append(f"k*={ks} {ts[ks-1]:5.2f}{'+' if on else '-'}")
    print(f"pi0={pi0:4.3f}: " + "  ".join(row))

# ---------------- break-even surfaces ----------------
print("\n=== break-even L_BE(k;a,b) = (a+b d(k))/S  [tokens/round incl. correction] ===")
for a in (17, 60, 100, 150):
    for b in (0.6, 0.95, 1.22):
        vals = [C(k, a, b)/S for k in (2, 4, 7, 8)]
        print(f"a={a:3d} b={b:.2f}: k2={vals[0]:5.3f} k4={vals[1]:5.3f} k7={vals[2]:5.3f} k8={vals[3]:5.3f}")
# reverse-engineer the 3.6 claim
for a in (17, 60, 100):
    bstar = (3.6 * S - a) / d[7]
    print(f"audit 3.6@k7 needs b = {bstar:.3f} ms/record (a={a})")

print("\n=== spec-ON threshold at k=2 with fallback f: q1 < (f - (b*d2 - S... )) ===")
for f in (0.0, 450.0, 500.0, 670.0):
    # T(2) > 1/S :  (1+q1)/(a+b*d2+f*(1-q1)) > 1/S  with a=100,b=0.6
    a, b = 100.0, 0.6
    # S*(1+q1) > a+b*d2+f*(1-q1)  ->  q1*(S+f) > a+b*d2+f-S
    thr = (a + b*d[2] + f - S) / (S + f)
    print(f"f={f:5.0f} ms: OFF iff q1 < {thr:.3f}")

# ---------------- sequential vs batch ----------------
print("\n=== sequential mode (a_t dense per position, b same) ===")
at_options = (250.0, 369.3)
for name, q in regimes.items():
    Ls = [L_of(q, k) for k in range(1, 9)]
    tb = T_table(q, 100, 0.6)
    kb = argmax_k(tb, 2)
    for at in at_options:
        best_seq = 0.0; kbest = 0
        for k in range(2, 9):
            cost = at + 0.6*d[1]
            for j in range(1, k):
                cost += q[j-1] * (at + 0.6*ddrec[j+1])
            t = Ls[k-1] / cost * 1000
            if t > best_seq: best_seq, kbest = t, k
        print(f"{name:32s} a_t={at:5.0f}: batch k*={kb} {tb[kb-1]:.2f} t/s | seq best k={kbest} {best_seq:.2f} t/s")
    # threshold a_t where seq matches batch at its k
    k = kb
    at_thr = 0.6*d[k] - 0.6*d[1] - sum(q[j-1]*0.6*ddrec[j+1] for j in range(1, k))
    at_thr = at_thr / max(1e-9, sum(q[j-1] for j in range(1, k)))
    print(f"{name:32s} seq==batch at a_t = {at_thr:.0f} ms")

# ---------------- heuristic loss ----------------
print("\n=== current heuristic k=clamp(int(1.3*mu)+1,2,8): fixed point & loss ===")
for name, q in regimes.items():
    Ls = [L_of(q, k) for k in range(1, 9)]
    ts = T_table(q, 100, 0.6)
    ks = argmax_k(ts, 2)
    k = 4
    for _ in range(20):
        k = max(2, min(8, int(1.3 * Ls[k-1]) + 1))
    loss = (ts[ks-1] / ts[k-1] - 1) * 100
    print(f"{name:32s} heuristic k={k} ({ts[k-1]:.3f} t/s) vs k*={ks} ({ts[ks-1]:.3f} t/s)  loss={loss:.1f}%")

# ---------------- campaign calibration vs measured best ----------------
print("\n=== calibration vs measured best 187.7-194.4 ms/token ===")
qc = regimes["campaign-meas (L4=3.70,L7=5.88)"]
for a in (60, 100, 150):
    ts = T_table(qc, a, 0.6)
    print(f"a={a}: T(6)={1000/ts[5]:.1f} ms/tok  T(7)={1000/ts[6]:.1f} ms/tok  T(8)={1000/ts[7]:.1f} ms/tok")
print("round-wall implied b at k=7: (1.0s..1.23s - a)/d7:",
      [round((w-100)/d[7], 3) for w in (990, 1100, 1230)])

# sliding survival bar
print("\n=== sliding bar: q_k > T* * b * Dd_k ===")
for Tstar in (2.8, 4.0, 5.3):
    bar = [Tstar/1000.0 * 0.6 * ddrec[k] for k in range(2, 9)]
    print(f"T*={Tstar} tok/s: bar per position 2..8 = {[round(x,3) for x in bar]}")
