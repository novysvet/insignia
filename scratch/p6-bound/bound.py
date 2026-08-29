#!/usr/bin/env python3
# P6: definitive decode throughput bound, glm-box (RTX 4070 Ti SUPER 16GB, 1x NVMe, WSL2).
# All bandwidths decimal GB/s (measured raw). Record sizes in MiB (binary).
MiB = 1024 ** 2
GiB = 1024 ** 3
GB = 1e9

R_FULL = 13.56 * MiB          # current NVFP4 expert record, bytes
R_PACK = 12.75 * MiB          # scale-packed lever, bytes
B_PCIE = 23.2 * GB            # pinned H2D raw
B_VRAM = 800 * GB             # GDDR6 observed
NVME = {"achieved": 3.7 * GB, "glmbox_typ": 4.7 * GB,
        "steady_lo": 5.45 * GB, "steady_hi": 5.84 * GB}

# ---- union curve U(K): distinct experts/layer over K consecutive tokens ----
U_MEAS = {1: 8.0, 2: 14.45, 3: 20.61, 4: 26.40, 5: 31.40}
# central extrapolation: power-law fit of measured increments inc(K)=6.45*(K/2)^-0.278
def U(K, band="central"):
    if K <= 5:  # linear interp on measured curve for fractional K
        if K in U_MEAS:
            return U_MEAS[K]
        k0 = int(K)
        return U_MEAS[k0] + (K - k0) * (U_MEAS[k0 + 1] - U_MEAS[k0])
    if band == "central":
        u = U_MEAS[5]
        for k in range(6, int(K) + 1):
            u += 6.45 * (k / 2) ** -0.278
        frac = K - int(K)
        if frac:
            u += frac * 6.45 * ((int(K) + 1) / 2) ** -0.278
        return u
    if band == "low":    # linear fit of ratios r(K)=0.843-0.0388*(K-3.5)
        return 8 * K * (0.843 - 0.0388 * (K - 3.5))
    return 8 * K * (0.7241 + 0.3707 / K)  # high: 1/K ratio fit

def ratio(K, band="central"):
    return U(K, band) / (8 * K)

# per-committed-token record count; tail-skip => records=42*U(min(A,8)), batch => 42*U(k)
def n_records(A, mode="tail", k=8, band="central"):
    eff = min(A, 8) if mode == "tail" else k
    return 42 * U(eff, band) / A

# ---- compute floor per ROUND (batch verify, 2K ctx) ----
DENSE_FP8 = 8.13 * GiB              # dense FP8 cache (incl. shared experts, lm_head)
def c_round_ms(A, ctx=2048, mode="batch"):
    drafter = 17.0                                   # measured
    dense_stream = DENSE_FP8 / B_VRAM * 1e3          # weights VRAM->SM once per round (batch)
    if mode == "seq":
        dense_stream *= min(A, 8)
    expert_stream = 42 * U(min(A, 8)) * R_PACK / B_VRAM * 1e3  # union bytes VRAM->SM once
    # absorbed MLA attention FLOP: q*keys*64heads*1024*11 layers (P12 formula), ~120 TFLOPS eff
    attn = min(A, 8) * 64 * 1024 * ctx * 11 / 120e12 * 1e3
    return drafter + dense_stream + expert_stream + attn

def tokps(A, f_v, f_h, R=R_PACK, B_d=None, mode="tail", k=8, ctx=2048, c_mode=None, band="central"):
    B_d = B_d or NVME["steady_hi"]
    c_mode = c_mode or ("seq" if mode == "tail" else "batch")
    n = 42 * U(min(A, 8) if mode == "tail" else k, band) / A
    t_pcie = n * (1 - f_v) * R / B_PCIE            # ALL off-VRAM records H2D
    t_nvme = n * (1 - f_v) * (1 - f_h) * R / B_d   # disk-sourced records only
    t_comp = c_round_ms(A, ctx, c_mode) / 1e3 / A   # seconds per committed token
    t = max(t_pcie, t_nvme, t_comp)
    return {"n": n, "GiB_tok": n * R / GiB, "pcie_ms": t_pcie * 1e3,
            "nvme_ms": t_nvme * 1e3, "comp_ms": t_comp * 1e3,
            "T_ms": t * 1e3, "tokps": 1 / t,
            "bind": ["pcie", "nvme", "comp"][[t_pcie, t_nvme, t_comp].index(t)]}

def row(name, **kw):
    r = tokps(**kw)
    print(f"{name:<44} A={kw['A']:<4} f_v={kw['f_v']:<5} f_h={kw['f_h']:<5}"
          f" n={r['n']:6.1f}  pcie={r['pcie_ms']:7.1f}  nvme={r['nvme_ms']:7.1f}"
          f"  comp={r['comp_ms']:5.1f}  T={r['T_ms']:7.1f}ms  {r['tokps']:5.2f} tok/s  [{r['bind']}]")

print("== U(K) extrapolation ==")
for K in (2, 3, 4, 5, 6, 7, 8, 16):
    lo, c, hi = U(K, "low"), U(K), U(K, "high")
    tag = "meas" if K <= 5 else "extr"
    print(f"K={K:2d} [{tag}] U={c:6.2f} (band {lo:6.2f}..{hi:6.2f})  ratio={ratio(K):.3f}"
          f"  (band {ratio(K,'low'):.3f}..{ratio(K,'high'):.3f})")

print("\n== n(A) records/committed-token (tail-skip, central U) ==")
for A in (2.64, 4, 4.6, 5, 6, 7, 8):
    print(f"A={A:<5} n={n_records(A):6.1f} rec/tok  {n_records(A)*R_PACK/GiB:5.2f} GiB/tok (12.75MiB)"
          f"  {n_records(A)*R_FULL/GiB:5.2f} GiB/tok (13.56MiB)")

print("\n== single-channel caps at A=8 (n=%.1f rec/tok) ==" % n_records(8))
n8 = n_records(8)
for nm, R in (("13.56MiB", R_FULL), ("12.75MiB", R_PACK)):
    gb = n8 * R / GB
    print(f"{nm}: {gb:6.2f} GB/tok -> PCIe-only {gb/23.2*1e3:6.1f} ms ({23.2/gb:4.2f} tok/s)"
          f" | NVMe5.84 {gb/5.84*1e3:6.1f} ms ({5.84/gb:4.2f}) | NVMe5.45 {gb/5.45*1e3:6.1f} ms"
          f" ({5.45/gb:4.2f}) | NVMe3.7 {gb/3.7*1e3:6.1f} ms ({3.7/gb:4.2f})")

print("\n== validation vs measured ==")
row("scalar decode (meas 447-571 ms/tok)", A=1, f_v=0.032, f_h=0.80, R=R_FULL, B_d=NVME["glmbox_typ"])
row("today real text (meas ~2.0-2.2 tok/s)", A=2.64, f_v=0.032, f_h=0.30, R=R_FULL, B_d=NVME["glmbox_typ"], mode="batch", k=4)
row("today real text, tail-skip variant", A=2.64, f_v=0.032, f_h=0.30, R=R_FULL, B_d=NVME["glmbox_typ"])
row("today real text, f_h=0.5 eff (retention)", A=2.64, f_v=0.032, f_h=0.50, R=R_FULL, B_d=NVME["glmbox_typ"])
row("best-ever parrot (meas 5.15-5.33)", A=5.88, f_v=0.032, f_h=0.75, R=R_FULL, B_d=NVME["glmbox_typ"], mode="batch", k=7)
row("best-ever parrot, f_h=0.8 BW5.45", A=5.88, f_v=0.032, f_h=0.80, R=R_FULL, B_d=NVME["steady_lo"], mode="batch", k=7)
row("best-ever, f_h=1 clairvoyant host", A=5.88, f_v=0.032, f_h=0.999, R=R_FULL, B_d=NVME["glmbox_typ"], mode="batch", k=7)

print("\n== A-grid x coverage (R=12.75MiB, NVMe 5.84 steady, tail-skip, 2K ctx) ==")
for f_v, f_h in ((0.30, 0.31), (0.35, 0.40), (0.41, 0.45), (0.35, 0.80)):
    for A in (5, 6, 7, 8):
        row(f"f_v={f_v} f_h={f_h}", A=A, f_v=f_v, f_h=f_h)
row("clairvoyant host (box ceiling)", A=8, f_v=0.41, f_h=0.999)
row("clairvoyant host, U-low band", A=8, f_v=0.41, f_h=0.999, band="low")
row("clairvoyant host, U-high band", A=8, f_v=0.41, f_h=0.999, band="high")
row("real-text acceptance A=4.6, realistic cov", A=4.6, f_v=0.32, f_h=0.36, B_d=NVME["steady_lo"])
row("real-text A=4.6, 2nd drive 11.68", A=4.6, f_v=0.32, f_h=0.36, B_d=11.68 * GB)
row("A=8 realistic cov, NVMe 5.45", A=8, f_v=0.35, f_h=0.40, B_d=NVME["steady_lo"])
row("A=8 realistic cov, NVMe 3.7 ach", A=8, f_v=0.35, f_h=0.40, B_d=NVME["achieved"])
row("A=8 realistic cov, 2nd drive", A=8, f_v=0.35, f_h=0.40, B_d=11.68 * GB)
row("A=8 in-sample cov, 2nd drive", A=8, f_v=0.41, f_h=0.45, B_d=11.68 * GB)

print("\n== compute floor sensitivity (ms/round, ms/committed-token at A=8) ==")
for ctx in (2048, 8192, 16384, 32768):
    c = c_round_ms(8, ctx)
    print(f"ctx={ctx:6d}: C_round={c:7.1f} ms -> {c/8:6.1f} ms/tok at A=8  (cap {1000/(c/8):5.1f} tok/s)")

print("\n== 20 tok/s requirement surface (R=12.75MiB, A=8: n*R = %.3f GB) ==" % (n8 * R_PACK / GB))
nR = n8 * R_PACK
for bd_nm, bd in (("5.84", 5.84e9), ("11.68(2 drives)", 11.68e9)):
    need_pc = 1 - 0.05 * B_PCIE / nR
    for f_h in (0.45, 0.60, 0.75, 0.999):
        need = 1 - 0.05 * bd / nR / (1 - f_h)
        print(f"disk {bd_nm:>15}: f_v >= {need_pc:.3f} (PCIe, always) | disk side f_v >= {max(need, 0):.3f} at f_h={f_h}")
print("\nrequired VRAM slots/bytes for given f_v (in-sample log-interp top-8 41.07%%, top-28 91.49%%):")
import math
def topk_for_cov(cov):
    return 8 * math.exp((cov - 0.4107) / (0.9149 - 0.4107) * math.log(28 / 8))
for cov in (0.35, 0.41, 0.54, 0.57, 0.63, 0.70):
    k = topk_for_cov(cov)
    slots = k * 42
    print(f"f_v={cov:.2f}: top-{k:.1f}/layer, {slots:.0f} slots, {slots*R_PACK/GiB:5.2f} GiB (12.75MiB)"
          f"  {slots*R_FULL/GiB:5.2f} GiB (13.56MiB)")
print("\nVRAM budget: 16 GiB = %.2f GB; dense 8.13GiB=%.2fGB, drafter 1.07GiB=%.2fGB,"
      " latents8K 1.4GiB=%.2fGB, ctx+scratch ~1.0GB" % (16 * GiB / GB, 8.13 * GiB / GB, 1.07 * GiB / GB, 1.4 * GiB / GB))
free = 16 * GiB / GB - 8.13 * GiB / GB - 1.07 * GiB / GB - 1.4 * GiB / GB - 1.0
print("free for expert tier: %.2f GB = %.2f GiB -> %.0f slots (12.75MiB), %.0f slots (13.56MiB)"
      % (free, free / (GiB / GB), free / (R_PACK / GB), free / (R_FULL / GB)))

print("\n== hardware scenarios ==")
row("16GB + 2nd drive + in-sample cov", A=8, f_v=0.41, f_h=0.45, B_d=11.68 * GB)
row("24GB card f_v=0.60 real, 1 drive", A=8, f_v=0.60, f_h=0.45)
row("24GB card f_v=0.60 real, 2 drives", A=8, f_v=0.60, f_h=0.55, B_d=11.68 * GB)
row("32GB card f_v=0.75, 1 drive f_h=.55", A=8, f_v=0.75, f_h=0.55)
row("32GB card f_v=0.75, 2 drives f_h=.62", A=8, f_v=0.75, f_h=0.62, B_d=11.68 * GB)
row("32GB real-text A=4.6 f_v=.70 2dr", A=4.6, f_v=0.70, f_h=0.55, B_d=11.68 * GB)
row("CPU-mixed: host bypasses PCIe (see notes)", A=8, f_v=0.35, f_h=0.62, B_d=11.68 * GB)

print("\n== A required for 20 tok/s at f_v=0.41 (in-sample ceiling), clairvoyant host ==")
for bd in (5.84e9, 11.68e9):
    # PCIe: (1-f_v)*42*U(A)/A*R <= 0.05*B_PCIE ; disk side with f_h=1 -> n/a
    for f_h in (0.999,):
        lim = 0.05 * B_PCIE / ((1 - 0.41) * R_PACK)
        print(f"disk {bd/1e9:.2f}: need 42*U(A)/A <= {lim:.1f} rec/tok (PCIe side)")
        for A in (8, 12, 16, 20, 24, 32):
            nn = 42 * U(A) / A
            print(f"   A={A:2d}: n={nn:6.1f} {'OK' if nn <= lim else 'no'}")
