#include "insignia_glm53_index.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using insignia::glm53::ShardedIndex;
using insignia::glm53::TensorLocation;

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL_ROOT MODEL.index\n", argv[0]);
        return 64;
    }
    try {
        ShardedIndex model(argv[2], argv[1]);
        std::printf("GLM-5.3 index OK: %zu tensors, %zu shards, %.3f GiB\n",
                    model.tensor_count(), model.shard_count(), model.payload_bytes() / double(1ull << 30));
        std::printf("geometry: hidden=%u layers=%u vocab=%u experts=%u topk=%u moe=%u hc=%u\n",
                    model.hidden_size(), model.layers(), model.vocab_size(), model.experts(),
                    model.active_experts(), model.moe_intermediate(), model.hc_mult());

        const char *required[] = {
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.3.self_attn.q_a_proj.weight",
            "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
            "model.language_model.layers.44.self_attn.q_proj.weight",
            "model.language_model.layers.45.eh_proj.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        };
        for (const char *name : required) {
            const TensorLocation &tensor = model.tensor(name);
            std::printf("  %-76s shard=%03u off=%12llu bytes=%10llu\n", name, tensor.shard + 1,
                        static_cast<unsigned long long>(tensor.offset),
                        static_cast<unsigned long long>(tensor.bytes));
        }

        uint64_t expert_bytes = 0;
        int contiguous = 0;
        for (uint32_t expert = 0; expert < model.experts(); ++expert) {
            const std::string base = "model.language_model.layers.3.mlp.experts." + std::to_string(expert) + ".";
            const TensorLocation &down = model.tensor(base + "down_proj.weight");
            const TensorLocation &gate = model.tensor(base + "gate_proj.weight");
            const TensorLocation &up = model.tensor(base + "up_proj.weight");
            expert_bytes += down.bytes + gate.bytes + up.bytes;
            contiguous += down.shard == gate.shard && gate.shard == up.shard &&
                          down.offset + down.bytes == gate.offset && gate.offset + gate.bytes == up.offset;
        }
        std::printf("layer-3 expert bodies: %.3f GiB; contiguous triplets=%d/%u\n",
                    expert_bytes / double(1ull << 30), contiguous, model.experts());

        const TensorLocation &sample = model.tensor("model.language_model.layers.3.mlp.experts.0.gate_proj.weight");
        std::vector<uint8_t> bytes(sample.bytes);
        const auto begin = std::chrono::steady_clock::now();
        model.read(sample, bytes.data());
        const auto end = std::chrono::steady_clock::now();
        uint64_t hash = 1469598103934665603ull;
        for (uint8_t byte : bytes) hash = (hash ^ byte) * 1099511628211ull;
        const double seconds = std::chrono::duration<double>(end - begin).count();
        std::printf("sample read: %.2f MiB %.2f GB/s fnv=%016llx\n", sample.bytes / double(1 << 20),
                    sample.bytes / seconds / 1e9, static_cast<unsigned long long>(hash));
    } catch (const std::exception &error) {
        std::fprintf(stderr, "glm53-index-check: %s\n", error.what());
        return 1;
    }
    return 0;
}
