#!/usr/bin/env python3
"""F1 falsifier: replay campaign routing traces through candidate host-tier
policies BEFORE building the segment-LRU (s8 open-queue #2).

Access model: scalar decode visits (token, layer) in trace order; each visit
demands its 8 experts. A demand either hits the tier or inserts (evicting the
policy's victim when full). This mirrors load_batch: all 8 records of a
(layer, token) are touched together, then released.

Policies:
  lru        - plain LRU (current engine behavior reference)
  slru       - soft segment-LRU: insert to probationary; promote to protected
               on hit while probationary; protected soft-capped at 50% of
               capacity (overflow demotes LRU protected back to probationary);
               victim = probationary LRU else protected LRU
  slru-demote- slru plus round-level demotion: after each token, records read
               this token that are NOT destined to be read again next token...
               (placeholder: with scalar traces there is no verify union; the
               demote arm is exercised by inserting this token's records at
               probationary-MRU vs probationary-LRU)
"""
import glob
import sys
from collections import OrderedDict

CAPACITIES = (1024, 1819, 2425, 3000)


def load_stream(paths):
    """Yield (token, layer, (e0..e7)) rows in trace order."""
    for path in sorted(paths):
        with open(path) as handle:
            for line in handle:
                fields = line.split()
                if len(fields) < 10 or fields[0].startswith("#"):
                    continue
                yield (int(fields[0]), int(fields[1]),
                       tuple(int(x) for x in fields[2:10]))


class LruSim:
    def __init__(self, capacity):
        self.capacity = capacity
        self.map = OrderedDict()  # key -> None, MRU at end

    def access(self, keys):
        hits = 0
        for key in keys:
            if key in self.map:
                self.map.move_to_end(key)
                hits += 1
            else:
                if len(self.map) >= self.capacity:
                    self.map.popitem(last=False)
                self.map[key] = None
        return hits


class SlruSim:
    def __init__(self, capacity):
        self.capacity = capacity
        self.probation = OrderedDict()   # MRU at end
        self.protected = OrderedDict()   # MRU at end
        self.protected_cap = max(1, capacity // 2)

    def _demote_overflow(self):
        while len(self.protected) > self.protected_cap:
            key, _ = self.protected.popitem(last=False)
            self.probation[key] = None

    def _evict(self):
        if self.probation:
            self.probation.popitem(last=False)
        else:
            self.protected.popitem(last=False)

    def access(self, keys):
        hits = 0
        for key in keys:
            if key in self.protected:
                self.protected.move_to_end(key)
                hits += 1
            elif key in self.probation:
                # promote on first hit from probationary
                del self.probation[key]
                self.protected[key] = None
                self._demote_overflow()
                hits += 1
            else:
                if len(self.probation) + len(self.protected) >= self.capacity:
                    self._evict()
                self.probation[key] = None  # probationary-MRU insert
        return hits


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/var/lib/insignia/tracecampaign/traces"
    paths = sorted(glob.glob(f"{root}/*.txt"))
    print(f"{len(paths)} trace files from {root}")
    rows = list(load_stream(paths))
    print(f"{len(rows)} (token,layer) rows, {sum(8 for _ in rows) or len(rows)*8} record accesses")
    for policy_name, factory in (("lru", LruSim), ("slru", SlruSim)):
        for capacity in CAPACITIES:
            sim = factory(capacity)
            hits = total = 0
            for token, layer, experts in rows:
                keys = tuple((layer, expert) for expert in experts)
                hits += sim.access(keys)
                total += len(keys)
            print(f"{policy_name:>8} cap={capacity:5d}: hit {hits}/{total} "
                  f"= {100.0 * hits / total:.2f}%")
    # Note: scalar-trace harm-check only. The verify-union benefit case rests
    # on the insert-rate analysis (audits/s7 P7 + s8 host-tier re-derivation),
    # which needs moe_multi route traces as next evidence.


if __name__ == "__main__":
    main()
