#pragma once
#include <array>
#include <cstdint>

namespace insignia::ada_dispatch_certificate {
// Historical 4070 SUPER evidence. Remote table is intentionally absent.
constexpr std::array<std::uint8_t, 8> kGateUpPairWarps{8, 8, 8, 8, 8, 8, 8, 8};
constexpr std::array<std::uint8_t, 8> kDownStoreWarps{4, 4, 8, 4, 4, 4, 4, 4};
constexpr std::array<std::uint8_t, 8> kPackedDownStoreWarps{4, 4, 8, 4, 4, 4, 4, 4};
constexpr std::array<std::uint8_t, 8> kDownWeightedWarps{4, 4, 8, 8, 4, 4, 4, 4};
constexpr bool kRtx4070SuperTableCertified = false;
constexpr bool kRtx4070TiSuperTableCertified = false;
}  // namespace insignia::ada_dispatch_certificate
