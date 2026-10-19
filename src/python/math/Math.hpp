#pragma once

#include "math/MathCore.hpp"
#include "math/MathParsers.hpp"
#include <Python.h>

namespace CPH::Python {

// Python extension type exported by this binding unit.
extern PyType_Spec MathServiceSpec;

} // namespace CPH::Python
