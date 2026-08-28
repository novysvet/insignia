import sys
sys.path.insert(0, "/mnt/e/coding/Insignia/tools")
from stripe_repack import load_index
import pathlib

_, ashard, aent = load_index(pathlib.Path("/var/lib/insignia/glm53-flash-text.index"))
_, bshard, bent = load_index(pathlib.Path("/var/lib/insignia/glm53-flash-text-striped.index"))

a = {e[0]: e for e in aent}
b = {e[0]: e for e in bent}
assert len(a) == len(b)

expected_remapped = expected_same = 0
bad = []
for name, ea in a.items():
    eb = b[name]
    stripe_expected = False
    if ".experts." in name:
        layer = int(name.split(".layers.")[1].split(".")[0])
        expert = int(name.split(".mlp.experts.")[1].split(".")[0])
        stripe_expected = 3 <= layer <= 44 and expert % 10 < 3
    if stripe_expected:
        expected_remapped += 1
        if eb[3] < 120:
            bad.append(("NOT REMAPPED", name, None))
    else:
        expected_same += 1
        if (ea[3], ea[5], ea[6]) != (eb[3], eb[5], eb[6]):
            bad.append(("LOCATION CHANGED", name, ((ea[3], ea[5], ea[6]), (eb[3], eb[5], eb[6]))))

print(f"expected remapped {expected_remapped}, expected same {expected_same}")
print(f"bad: {len(bad)}")
for kind, name, loc in bad[:25]:
    print(kind, name, loc)
