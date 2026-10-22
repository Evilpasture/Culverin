// Type alias for C++ files

#pragma once
#include <cstddef>
#include <cstdint>
#ifdef _MSC_VER
#    define CPH_RESTRICT __restrict
#else
#    define CPH_RESTRICT __restrict__
#endif
namespace CPH {
using Unsigned8   = std::uint8_t;
using Unsigned32  = std::uint32_t;
using Unsigned64  = std::uint64_t;
using Integer8    = std::int8_t;
using Integer32   = std::int32_t;
using Integer64   = std::int64_t;
using Float32     = float;
using Float64     = double;
using ByteType    = std::uint8_t;
using SizeType    = std::size_t;
using OffsetType  = std::ptrdiff_t;
using AddressType = std::uintptr_t;
} // namespace CPH