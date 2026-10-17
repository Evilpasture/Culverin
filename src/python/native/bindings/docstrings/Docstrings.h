#pragma once

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Stitches Markdown docstrings directly into an array of PyMethodDef.
 * Searches the markdown string for:
 *   ## class <class_name>
 *   ### <method_name>
 *
 * @param methods The null-terminated array of PyMethodDef.
 * @param class_name The name of the class as defined in the Markdown headers.
 * @param markdown_content The null-terminated Markdown string.
 */
void culv_docs_stitch_methods(PyMethodDef *methods, const char *class_name, const char *markdown_content);

/**
 * Stitches Markdown docstrings directly into an array of PyGetSetDef.
 */
void culv_docs_stitch_getset(PyGetSetDef *getset, const char *class_name, const char *markdown_content);

/**
 * Iterates through a PyType_Spec and stitches Markdown docstrings into its
 * methods and getsets.
 */
void culv_docs_stitch_spec(PyType_Spec *spec, const char *class_name, const char *markdown_content);

#ifdef __cplusplus
}
#endif