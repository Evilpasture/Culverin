#include "culverin_fast_parse.h"
#include "culverin_compiler_specifics.h"

/**
 * fp_report_missing
 * Cold path: generates the Python TypeError for missing required arguments.
 */
bool fp_report_missing(const FastParser *fp, uint64_t provided_mask) {
  for (size_t i = 0; i < fp->count; i++) {
    if ((int)fp->specs[i].required && !(provided_mask & (1ULL << i))) {
      PyErr_Format(PyExc_TypeError, "required argument '%s' missing",
                   fp->specs[i].name);
      return false;
    }
  }
  return false;
}

bool fp_report_multiple(const FastParser *fp, size_t index) {
  PyErr_Format(PyExc_TypeError, "argument '%s' got multiple values",
               fp->specs[index].name);
  return false;
}

bool fp_report_too_many(const FastParser *fp, Py_ssize_t nargs) {
  PyErr_Format(PyExc_TypeError,
               "too many positional arguments (expected %zu, got %zd)",
               fp->count, nargs);
  return false;
}

/**
 * fp_init_impl
 * Initialization logic: intern strings and build O(1) lookup table.
 */
void fp_init_impl(FastParser *fp, FastArgSpec *specs, size_t count) {
  if (count > 64) {
    Py_FatalError("FastParse: Argument count exceeds bitmask limit of 64.");
  }

  fp->specs = specs;
  fp->count = count;
  fp->required_mask = 0;
  fp->lookup_table = NULL;

  for (size_t i = 0; i < count; i++) {
    if (specs[i].name) {
      specs[i].interned = PyUnicode_InternFromString(specs[i].name);
    }
    if (specs[i].required) {
      fp->required_mask |= (1ULL << i);
    }
  }

  if (count > 10) {
    size_t table_size = 1;
    while (table_size < (count * 2)) {
      table_size <<= 1;
    }
    fp->table_mask = table_size - 1;
    fp->lookup_table =
        (uint16_t *)PyMem_RawMalloc(table_size * sizeof(uint16_t));

    for (size_t i = 0; i < table_size; i++) {
      fp->lookup_table[i] = FP_EMPTY_SLOT;
    }

    for (size_t i = 0; i < count; i++) {
      size_t h = fp_hash_ptr(fp->specs[i].interned, fp->table_mask);
      while (fp->lookup_table[h] != FP_EMPTY_SLOT) {
        h = (h + 1) & fp->table_mask;
      }
      fp->lookup_table[h] = (uint16_t)i;
    }
  }
}

/**
 * fp_parse_legacy
 * Handle standard (PyObject *args, PyObject *kwargs) calling convention.
 * Not intended for hot loops, so we keep it out of the header.
 */
bool fp_parse_legacy(PyObject *args, PyObject *kwargs, const FastParser *fp,
                     void **targets, [[maybe_unused]] size_t dummy) {
  uint64_t provided_mask = 0;
  const size_t count = fp->count;
  const FastArgSpec *specs = fp->specs;

  if (args) {
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs > (Py_ssize_t)count)
      return fp_report_too_many(fp, nargs);

    for (Py_ssize_t i = 0; i < nargs; ++i) {
      provided_mask |= (1ULL << i);
      specs[i].convert(PyTuple_GET_ITEM(args, i), targets[i]);
    }
  }

  if (kwargs) {
    PyObject *key, *val;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwargs, &pos, &key, &val)) {
      int idx = -1;
      // Search for key
      for (size_t i = 0; i < count; ++i) {
        if (key == specs[i].interned ||
            PyUnicode_Compare(key, specs[i].interned) == 0) {
          idx = (int)i;
          break;
        }
      }

      if (idx == -1) {
        PyErr_Format(PyExc_TypeError, "unexpected keyword argument '%U'", key);
        return false;
      }
      if (provided_mask & (1ULL << idx)) {
        return fp_report_multiple(fp, (size_t)idx);
      }

      provided_mask |= (1ULL << idx);
      specs[idx].convert(val, targets[idx]);
    }
  }

  if (UNLIKELY((provided_mask & fp->required_mask) != fp->required_mask)) {
    return fp_report_missing(fp, provided_mask);
  }
  return (PyErr_Occurred() == NULL);
}