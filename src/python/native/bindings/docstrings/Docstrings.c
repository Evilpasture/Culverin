#include "Docstrings.h"
#include <stdio.h>
#include <string.h>

static const char *CULV_DOCS_COMMENT_MARKER = "<!--";

static char *culv_docs_allocate_docstring(const char *start, size_t length) {
    char *doc = (char *)PyMem_RawMalloc(length + 1);
    if (doc == nullptr) {
        return nullptr;
    }
    memcpy(doc, start, length);
    doc[length] = '\0';
    return doc;
}

// Master Docstring Extractor (Handles Nested ## class -> ### method)
static char *culv_docs_extract_docstring(const char *docs_ptr, const char *class_name,
                               const char *method_name) {
    if (!docs_ptr || !class_name || !method_name) {
        return nullptr;
    }

    constexpr size_t CULV_DOCS_BUFFER_SIZE = 128;
    char class_key[CULV_DOCS_BUFFER_SIZE];
    snprintf(class_key, sizeof(class_key), "## class %s", class_name);

    // 1. Find the Class boundary
    const char *class_start = strstr(docs_ptr, class_key);
    if (!class_start) {
        return nullptr;
    }

    // The scope of this class ends when the next class begins
    const char *class_end = strstr(class_start + 1, "## class ");

    // 2. Find the Method boundary within this class
    char method_key[CULV_DOCS_BUFFER_SIZE];
    snprintf(method_key, sizeof(method_key), "### %s", method_name);

    const char *method_start = strstr(class_start, method_key);

    // Ensure we found it AND it belongs to THIS class AND isn't a substring
    // (like finding "step_up" for "step")
    while (method_start && (class_end == nullptr || method_start < class_end)) {
        char c = *(method_start + strlen(method_key));
        // We allow '(', '\r', '\n', ' ', or '\0' after the method name
        if (c == '(' || c == '\r' || c == '\n' || c == ' ' || c == '\0') {
            break; // Valid match
        }
        method_start = strstr(method_start + 1, method_key); // Keep searching
    }

    if (!method_start || (class_end != nullptr && method_start >= class_end)) {
        return nullptr;
    }

    // 3. Move past the "### method(...)\n" header
    const char *doc_start = method_start;
    while (*doc_start != '\0' && *doc_start != '\n' && *doc_start != '\r') {
        doc_start++;
    }
    while (*doc_start == '\n' || *doc_start == '\r') {
        doc_start++;
    }

    // 4. Skip HTML Comments if present
    if (strncmp(doc_start, CULV_DOCS_COMMENT_MARKER, 4) == 0) {
        const char *comment_end = strstr(doc_start, "-->");
        if (comment_end) {
            doc_start = comment_end + 3; // Skip "-->"
            while (*doc_start == '\n' || *doc_start == '\r' || *doc_start == ' ') {
                doc_start++;
            }
        }
    }

    // 5. Find the end of this method's docs (next ### or ##)
    const char *doc_end = doc_start;
    while (*doc_end != '\0') {
        if (strncmp(doc_end, "### ", 4) == 0 || strncmp(doc_end, "## ", 3) == 0) {
            break;
        }
        doc_end++;
    }

    // 6. Trim trailing whitespace
    if (doc_end > doc_start) {
        doc_end--;
        while (doc_end > doc_start &&
               (*doc_end == '\n' || *doc_end == '\r' || *doc_end == ' ' || *doc_end == '\t')) {
            doc_end--;
        }
    }

    if (doc_end >= doc_start) {
        return culv_docs_allocate_docstring(doc_start, (size_t)(doc_end - doc_start + 1));
    }

    return nullptr;
}

void CPH_CulvDocsStitchMethods(PyMethodDef *methods, const char *class_name, const char *markdown_content) {
    if (methods == nullptr || markdown_content == nullptr) {
        return;
    }
    for (PyMethodDef *m = methods; m->ml_name != nullptr; m++) {
        if (m->ml_doc == nullptr) {
            m->ml_doc = culv_docs_extract_docstring(markdown_content, class_name, m->ml_name);
        }
    }
}

void CPH_CulvDocsStitchGetset(PyGetSetDef *getset, const char *class_name, const char *markdown_content) {
    if (getset == nullptr || markdown_content == nullptr) {
        return;
    }
    for (PyGetSetDef *g = getset; g->name != nullptr; g++) {
        if (g->doc == nullptr) {
            g->doc = culv_docs_extract_docstring(markdown_content, class_name, g->name);
        }
    }
}

void CPH_CulvDocsStitchSpec(PyType_Spec *spec, const char *class_name, const char *markdown_content) {
    if (!spec || !spec->slots || !markdown_content) {
        return;
    }
    // Cast to PyType_Slot* to iterate
    for (const PyType_Slot *slot = (const PyType_Slot *)spec->slots; slot->slot != 0; slot++) {
        if (slot->slot == Py_tp_methods) {
            CPH_CulvDocsStitchMethods((PyMethodDef *)slot->pfunc, class_name, markdown_content);
        } else if (slot->slot == Py_tp_getset) {
            CPH_CulvDocsStitchGetset((PyGetSetDef *)slot->pfunc, class_name, markdown_content);
        }
    }
}