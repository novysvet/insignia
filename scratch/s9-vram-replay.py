#!/usr/bin/env python3
"""Held-out replay of the production per-layer host/VRAM expert tiers.

The pin list is learned only from odd-numbered real benchmark traces and is
evaluated on even-numbered traces.  Device insertions use the production
per-layer segment geometry and sequential upload order.  A device pin count
must leave at least one recyclable slot in every layer segment.

This is the offline falsifier for INSIGNIA_GLM53_VRAM_COMPACT_SEGMENTS and
INSIGNIA_GLM53_VRAM_BATCH_VICTIM.  It expects the Session 9 merged trace and
manifest as positional arguments; those large data files are not committed.
"""

import argparse
import collections
import csv


EXPERTS = 288
LAYERS = tuple(range(3, 45))
TRAIN_RUNS = {f"p{index:02d}" for index in range(1, 12, 2)}
TEST_RUNS = {f"p{index:02d}" for index in range(2, 13, 2)}


def layer_caps(total, compact=False):
    segments = max(1, min(len(LAYERS) if compact else 46, total))
    result = {}
    for layer in LAYERS:
        segment = (layer - LAYERS[0] if compact else layer) % segments
        begin = segment * total // segments
        end = total if segment + 1 == segments else (segment + 1) * total // segments
        result[layer] = end - begin
    return result


def load_manifest(path):
    segments = []
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            base = int(row["token_base"])
            segments.append({
                "run": row["run"],
                "lo": base + 1,
                "hi": base + int(row["tokens"]),
            })
    return segments


def segment_for(token, segments):
    for index, segment in enumerate(segments):
        if segment["lo"] <= token <= segment["hi"]:
            return index
    return -1


def learn_pins(trace, segments):
    counts = {layer: collections.Counter() for layer in LAYERS}
    with open(trace, encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if len(fields) < 10:
                continue
            token, layer = int(fields[0]), int(fields[1])
            found = segment_for(token, segments)
            if found < 0 or segments[found]["run"] not in TRAIN_RUNS:
                continue
            counts[layer].update(int(value) for value in fields[2:10])
    return {
        layer: [expert for expert, _ in counts[layer].most_common()]
        for layer in LAYERS
    }


class Device:
    def __init__(self, slots, pins, compact, batch_aware):
        self.caps = layer_caps(slots, compact)
        self.pin_keys = pins
        self.batch_aware = batch_aware
        self.layers = {layer: collections.OrderedDict() for layer in LAYERS}

    def contains(self, key):
        return key in self.layers[key[0]]

    def touch(self, key, future):
        cache = self.layers[key[0]]
        if key in cache:
            if key not in self.pin_keys:
                cache.move_to_end(key)
            return True
        cap = self.caps[key[0]]
        if len(cache) >= cap:
            candidates = [candidate for candidate in cache if candidate not in self.pin_keys]
            victim = candidates[0] if candidates else None
            if self.batch_aware and candidates:
                future_at = {candidate: index for index, candidate in enumerate(future)}
                cold = [candidate for candidate in candidates if candidate not in future_at]
                victim = cold[0] if cold else max(candidates, key=future_at.__getitem__)
            if victim is None:
                raise RuntimeError(f"layer {key[0]} has no recyclable device slot")
            del cache[victim]
        cache[key] = None
        return False


class Host:
    def __init__(self, capacity, pins):
        if len(pins) >= capacity:
            raise ValueError("host pins consume the whole host tier")
        self.capacity = capacity
        self.pin_keys = pins
        self.resident = set(pins)
        self.dynamic = collections.OrderedDict()
        self.claimed = set()

    def contains(self, key):
        return key in self.resident

    def claim(self, key):
        if key in self.dynamic:
            self.claimed.add(key)

    def insert(self, key):
        if key in self.resident:
            self.claim(key)
            return
        while len(self.resident) >= self.capacity:
            victim = next((candidate for candidate in self.dynamic
                           if candidate not in self.claimed), None)
            if victim is None:
                raise RuntimeError("all host records are claimed")
            del self.dynamic[victim]
            self.resident.remove(victim)
        self.resident.add(key)
        self.dynamic[key] = None
        self.claimed.add(key)

    def touch(self, key):
        if key in self.dynamic:
            self.dynamic.move_to_end(key)
        self.claimed.discard(key)


class Simulation:
    def __init__(self, host_slots, device_slots, rankings, host_pin, device_pin,
                 compact=False, batch_aware=False, f3=True):
        self.host_slots = host_slots
        self.device_slots = device_slots
        self.host_pins = {
            (layer, expert)
            for layer in LAYERS for expert in rankings[layer][:host_pin]
        }
        self.device_pins = {
            (layer, expert)
            for layer in LAYERS for expert in rankings[layer][:device_pin]
        }
        self.compact = compact
        self.batch_aware = batch_aware
        caps = layer_caps(device_slots, compact)
        bad = [layer for layer in LAYERS
               if sum(key[0] == layer for key in self.device_pins) >= caps[layer]]
        if bad:
            raise ValueError(f"device pins leave no recyclable slot in layers {bad}")
        self.f3 = f3
        self.requests = self.host_hits = self.device_hits = 0
        self.nvme = self.h2d = self.tokens = 0
        self.reset()

    def reset(self):
        self.host = Host(self.host_slots, self.host_pins)
        self.device = Device(self.device_slots, self.device_pins, self.compact,
                             self.batch_aware)

    def group(self, layer, experts):
        keys = list(dict.fromkeys((layer, expert) for expert in experts))
        self.requests += len(keys)
        host_present = [self.host.contains(key) for key in keys]
        for key, present in zip(keys, host_present):
            if present:
                self.host.claim(key)
        device_adopt = [
            not present and self.f3 and self.device.contains(key)
            for key, present in zip(keys, host_present)
        ]
        for index, key in enumerate(keys):
            if host_present[index]:
                self.host_hits += 1
            elif not device_adopt[index]:
                self.host.insert(key)
                self.nvme += 1

        for index, key in enumerate(keys):
            if device_adopt[index] and not self.device.contains(key):
                self.host.insert(key)
                self.nvme += 1
                device_adopt[index] = False
            if self.device.touch(key, keys[index + 1:]):
                self.device_hits += 1
            else:
                self.h2d += 1
            if self.host.contains(key):
                self.host.touch(key)


def replay(trace, segments, simulations):
    active_segment = -1
    token = None
    rows = []

    def finish():
        if len(rows) != len(LAYERS) or {layer for layer, _ in rows} != set(LAYERS):
            return
        for simulation in simulations:
            simulation.tokens += 1
            for layer, experts in rows:
                simulation.group(layer, experts)

    with open(trace, encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if len(fields) < 10:
                continue
            row_token, layer = int(fields[0]), int(fields[1])
            found = segment_for(row_token, segments)
            if found < 0 or segments[found]["run"] not in TEST_RUNS:
                continue
            if row_token != token:
                if token is not None:
                    finish()
                token, rows = row_token, []
            if found != active_segment:
                for simulation in simulations:
                    simulation.reset()
                active_segment = found
            rows.append((layer, [int(value) for value in fields[2:10]]))
    if token is not None:
        finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("manifest")
    parser.add_argument("--host", type=int, default=2425)
    parser.add_argument("--device", type=int, default=211)
    args = parser.parse_args()
    segments = load_manifest(args.manifest)
    rankings = learn_pins(args.trace, segments)
    configurations = [
        ("prod", 0, 0, False, False),
        ("batch", 0, 0, False, True),
        ("compact", 0, 0, True, False),
        ("compact-batch", 0, 0, True, True),
        ("compact-batch-pin1", 8, 1, True, True),
        ("compact-batch-pin2", 8, 2, True, True),
        ("compact-batch-pin3", 8, 3, True, True),
        ("compact-batch-pin4", 8, 4, True, True),
    ]
    simulations = [
        Simulation(args.host, args.device, rankings, host_pin, device_pin,
                   compact, batch_aware)
        for _, host_pin, device_pin, compact, batch_aware in configurations
    ]
    replay(args.trace, segments, simulations)
    print("policy,host_pin,device_pin,tokens,nvme_per_token,h2d_per_token,host_hit_pct,device_hit_pct")
    for (name, host_pin, device_pin, _, _), simulation in zip(configurations, simulations):
        print(f"{name},{host_pin},{device_pin},{simulation.tokens},"
              f"{simulation.nvme / simulation.tokens:.6f},"
              f"{simulation.h2d / simulation.tokens:.6f},"
              f"{100 * simulation.host_hits / simulation.requests:.4f},"
              f"{100 * simulation.device_hits / simulation.requests:.4f}")


if __name__ == "__main__":
    main()
