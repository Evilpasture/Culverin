#include <Python.h>
#include "debug_render.h"

// --- Debug Buffer Helpers ---
void debug_buffer_ensure(DebugBuffer* buf, size_t count_needed) {
    if (buf->count + count_needed > buf->capacity) {
        size_t new_cap = (buf->capacity == 0) ? 4096 : buf->capacity * 2;
        while (buf->count + count_needed > new_cap) new_cap *= 2;
        
        void* new_ptr = PyMem_RawRealloc(buf->data, new_cap * sizeof(DebugVertex));
        if (!new_ptr) return; // Silent fail on OOM for debug info
        
        buf->data = (DebugVertex*)new_ptr;
        buf->capacity = new_cap;
    }
}

void debug_buffer_push(DebugBuffer* buf, DebugCoordinates pos, uint32_t color) {
    if (buf->count >= buf->capacity) return; // Safety
    buf->data[buf->count].x = pos.x;
    buf->data[buf->count].y = pos.y;
    buf->data[buf->count].z = pos.z;
    buf->data[buf->count].color = color;
    buf->count++;
}

void debug_buffer_free(DebugBuffer* buf) {
    if (buf->data) PyMem_RawFree(buf->data);
    buf->data = NULL;
    buf->count = 0;
    buf->capacity = 0;
}
