#!/usr/bin/env python3
"""U5 evidence: cross-prompt routing shift via split-sample hot-set coverage.

Per TRACE-FORMAT §4 protocol: train arm = odd real-text runs, test = even
runs (p00/legacy excluded as atypically repetitive). Measures:
  1. per-layer empirical access distribution on train
  2. static top-B coverage in-sample (train) vs out-of-sample (test), B swept
  3. dynamic LRU reference on the same test prompts for comparison
The in/out gap prices the permanent tax on every static cache (audits/s7 U5).
"""
import glob
import sys
from collections import Counter, OrderedDict

def load_runs(root):
    runs = {}
    for path in sorted(glob.glob(f"{root}/*.txt")):
        rows = []
        with open(path) as handle:
            for line in handle:
                fields = line.split()
                if len(fields) < 10 or fields[0].startswith("#"):
                    continue
                rows.append((int(fields[1]), tuple(int(x) for x in fields[2:10])))
        if rows:
            runs[path.rsplit("/", 1)[-1]] = rows
    return runs

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/var/lib/insignia/tracecampaign/traces"
    runs = load_runs(root)
    names = sorted(runs)
    print(f"{len(names)} runs: {', '.join(names)}")
    train_names = names[1::2]
    test_names = names[0::2]
    print(f"train: {len(train_names)} runs, test: {len(test_names)} runs")

    def counter_for(subset):
        counts = {}
        for name in subset:
            for layer, experts in runs[name]:
                bucket = counts.setdefault(layer, Counter())
                for expert in experts:
                    bucket[expert] += 1
        return counts

    train_counts = counter_for(train_names)
    test_counts = counter_for(test_names)

    budgets = (8, 28, 57, 84)
    layers = sorted(set(train_counts) | set(test_counts))
    print(f"{len(layers)} sparse layers")
    header = "B  in-sample  out-sample  gap_pp  test-entropy_bits"
    print(header)
    import math
    for budget in budgets:
        ins = out = 0
        tot = 0
        ent_sum = 0.0
        for layer in layers:
            train_c = train_counts.get(layer, Counter())
            test_c = test_counts.get(layer, Counter())
            total_train = sum(train_c.values())
            total_test = sum(test_c.values())
            if not total_test:
                continue
            top = {expert for expert, _ in train_c.most_common(budget)}
            ins += sum(c for expert, c in train_c.items() if expert in top) / max(1, total_train)
            out += sum(c for expert, c in test_c.items() if expert in top) / total_test
            tot += 1
            ps = [c / total_test for c in test_c.values()]
            ent_sum += -sum(p * math.log2(p) for p in ps)
        ins_f = ins / tot
        out_f = out / tot
        print(f"{budget:3d}  {100.0*ins_f:8.2f}  {100.0*out_f:8.2f}  {100.0*(ins_f-out_f):7.2f}  {ent_sum/tot:5.2f}")

    # Dynamic LRU on the test prompts, per-prompt cold start under each arm
    # (this is what a static list competes against after warm-up).
    for capacity in (1024, 2425):
        for arm, subset in (("train", train_names), ("test", test_names)):
            hits = total = 0
            state = OrderedDict()
            for name in subset:
                for layer, experts in runs[name]:
                    for expert in experts:
                        key = (layer, expert)
                        if key in state:
                            state.move_to_end(key)
                            hits += 1
                        else:
                            if len(state) >= capacity:
                                state.popitem(last=False)
                            state[key] = None
                        total += 1
            print(f"lru cap={capacity} on {arm}: {100.0*hits/total:.2f}%")

if __name__ == "__main__":
    main()
