#pragma once

// Typed argument parsers (Phase 2, step B3).
//
// The pre-B3 API was `int ParseVec3F32(PyObject *obj, float *x, float *y,
// float *z)`: an int that means "1 ok / 0 failed", out-parameters, and a
// Python exception left in the thread state for the caller to notice. The
// replacements return `ParseResult` (see ParseError.hpp) and take `Borrowed`,
// which makes ownership explicit at every call site and lets the failure carry
// the diagnostic instead of hiding it in global state.
//
// The `_into` forms still write through references on purpose: the fast-parse
// trait table (ParserTraits.hpp) writes directly into the caller's argument
// slots, and the whole point of that table is to avoid copies on the hot path.

#include "bindings/PythonCommon.hpp"
#include "bindings/parsing/ParseError.hpp"
#include "core/Types.hpp"
#include "physics/PhysicsParams.hpp"

#include <expected>

namespace CPH::Python {

// --- Sequence-valued arguments: (x, y, z) / (x, y, z, w) -------------------
// Accepts any buffer-protocol object of the exact size, then falls back to a
// sequence of exactly `count` numbers.

[[nodiscard]] ParseResult ParseVec3Into(Borrowed obj, float &x, float &y, float &z);
[[nodiscard]] ParseResult ParseVec3Into(Borrowed obj, double &x, double &y, double &z);
[[nodiscard]] ParseResult ParseQuatInto(Borrowed obj, float &x, float &y, float &z, float &w);
[[nodiscard]] ParseResult ParseQuatInto(Borrowed obj, double &x, double &y, double &z,
                                          double &w);

// Stride types used by the shadow buffers.
[[nodiscard]] ParseResult ParseVec3Into(Borrowed obj, JPH::Float3 &out);
[[nodiscard]] ParseResult ParseVec3Into(Borrowed obj, PosStride &out);
[[nodiscard]] ParseResult ParseQuatInto(Borrowed obj, AuxStride &out);

// Value forms for call sites that want the result rather than an out-param.
[[nodiscard]] std::expected<JPH::Float3, ParseError> ParseVec3(Borrowed obj);
[[nodiscard]] std::expected<PosStride, ParseError> ParsePosition(Borrowed obj);
[[nodiscard]] std::expected<AuxStride, ParseError> ParseRotation(Borrowed obj);

// --- Keyword/dictionary helpers -------------------------------------------
// `ParseMotorConfig` and `ParseBodySize` cannot fail (missing keys fall back to
// defaults), so they keep their signatures; they only lose the raw pointer.

void ParseMotorConfig(Borrowed motor_dict, ConstraintParams &params);
void ParseBodySize(Borrowed py_size, float size[4]);

// --- Constraint parameter tuples ------------------------------------------
// These wrap PyArg_ParseTuple; the tuple-format strings are part of the
// Python-visible contract.

[[nodiscard]] ParseResult ParsePointParams(Borrowed args, ConstraintParams &params);
[[nodiscard]] ParseResult ParseHingeParams(Borrowed args, ConstraintParams &params);
[[nodiscard]] ParseResult ParseSliderParams(Borrowed args, ConstraintParams &params);
[[nodiscard]] ParseResult ParseConeParams(Borrowed args, ConstraintParams &params);
[[nodiscard]] ParseResult ParseDistanceParams(Borrowed args, ConstraintParams &params);

// --- Tracked-vehicle tracks ------------------------------------------------
// `out_data` must hold TrackData for every entry in `py_tracks`; the parsed
// count is returned alongside it.
struct ParsedTracks {
    int count = 0;
};

[[nodiscard]] std::expected<ParsedTracks, ParseError> ParseTracksToC(Borrowed py_tracks,
                                                                     TrackData *out_data);

} // namespace CPH::Python
