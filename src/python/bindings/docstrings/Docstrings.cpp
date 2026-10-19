#include "Docstrings.h"
#include "bindings/PythonCommon.hpp"

#include <cstdio>
#include <cstring>

namespace CPH::Python::detail {

inline constexpr char kDocsCommentMarker[] = "<!--";
inline constexpr std::size_t kDocsBufferSize = 128;

[[nodiscard]] char *AllocateDocstring(const char *start, std::size_t length) noexcept {
    auto *doc = static_cast<char *>(CPH::Python::Allocate(length + 1));
    if (doc == nullptr) {
        return nullptr;
    }
    std::memcpy(doc, start, length);
    doc[length] = '\0';
    return doc;
}

// Locate and copy the documentation for a method nested under one class heading.
[[nodiscard]] char *ExtractDocstring(const char *docs, const char *class_name,
                                     const char *method_name) noexcept {
    if (docs == nullptr || class_name == nullptr || method_name == nullptr) {
        return nullptr;
    }

    char class_key[kDocsBufferSize];
    std::snprintf(class_key, sizeof(class_key), "## class %s", class_name);

    const char *class_start = std::strstr(docs, class_key);
    if (class_start == nullptr) {
        return nullptr;
    }

    const char *class_end = std::strstr(class_start + 1, "## class ");

    char method_key[kDocsBufferSize];
    std::snprintf(method_key, sizeof(method_key), "### %s", method_name);

    const char *method_start = std::strstr(class_start, method_key);
    while (method_start != nullptr && (class_end == nullptr || method_start < class_end)) {
        const char next = *(method_start + std::strlen(method_key));
        if (next == '(' || next == '\r' || next == '\n' || next == ' ' || next == '\0') {
            break;
        }
        method_start = std::strstr(method_start + 1, method_key);
    }

    if (method_start == nullptr || (class_end != nullptr && method_start >= class_end)) {
        return nullptr;
    }

    const char *doc_start = method_start;
    while (*doc_start != '\0' && *doc_start != '\n' && *doc_start != '\r') {
        ++doc_start;
    }
    while (*doc_start == '\n' || *doc_start == '\r') {
        ++doc_start;
    }

    if (std::strncmp(doc_start, kDocsCommentMarker, 4) == 0) {
        const char *comment_end = std::strstr(doc_start, "-->");
        if (comment_end != nullptr) {
            doc_start = comment_end + 3;
            while (*doc_start == '\n' || *doc_start == '\r' || *doc_start == ' ') {
                ++doc_start;
            }
        }
    }

    const char *doc_end = doc_start;
    while (*doc_end != '\0') {
        if (std::strncmp(doc_end, "### ", 4) == 0 || std::strncmp(doc_end, "## ", 3) == 0) {
            break;
        }
        ++doc_end;
    }

    if (doc_end > doc_start) {
        --doc_end;
        while (doc_end > doc_start &&
               (*doc_end == '\n' || *doc_end == '\r' || *doc_end == ' ' || *doc_end == '\t')) {
            --doc_end;
        }
    }

    if (doc_end >= doc_start) {
        return AllocateDocstring(doc_start,
                                 static_cast<std::size_t>(doc_end - doc_start + 1));
    }
    return nullptr;
}

} // namespace CPH::Python::detail

extern "C" {

void CPH_DocsStitchMethods(PyMethodDef *methods, const char *class_name,
                           const char *markdown_content) {
    if (methods == nullptr || markdown_content == nullptr) {
        return;
    }
    for (PyMethodDef *method = methods; method->ml_name != nullptr; ++method) {
        if (method->ml_doc == nullptr) {
            method->ml_doc = CPH::Python::detail::ExtractDocstring(
                markdown_content, class_name, method->ml_name);
        }
    }
}

void CPH_DocsStitchGetset(PyGetSetDef *getset, const char *class_name,
                          const char *markdown_content) {
    if (getset == nullptr || markdown_content == nullptr) {
        return;
    }
    for (PyGetSetDef *property = getset; property->name != nullptr; ++property) {
        if (property->doc == nullptr) {
            property->doc = CPH::Python::detail::ExtractDocstring(
                markdown_content, class_name, property->name);
        }
    }
}

void CPH_DocsStitchSpec(PyType_Spec *spec, const char *class_name,
                        const char *markdown_content) {
    if (spec == nullptr || spec->slots == nullptr || markdown_content == nullptr) {
        return;
    }
    for (const PyType_Slot *slot = spec->slots; slot->slot != 0; ++slot) {
        if (slot->slot == Py_tp_methods) {
            CPH_DocsStitchMethods(static_cast<PyMethodDef *>(slot->pfunc), class_name,
                                  markdown_content);
        } else if (slot->slot == Py_tp_getset) {
            CPH_DocsStitchGetset(static_cast<PyGetSetDef *>(slot->pfunc), class_name,
                                 markdown_content);
        }
    }
}

} // extern "C"
