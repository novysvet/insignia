import sys
sys.path.insert(0, "/mnt/e/coding/Insignia/tools")
from glm53_layout_analysis import load_index
import pathlib
shards, entries = load_index(pathlib.Path("/var/lib/insignia/glm53-flash-text.index"))
names = [e[0] for e in entries]
print([n for n in names if "expert" in n][:2])
print(names[:2])
