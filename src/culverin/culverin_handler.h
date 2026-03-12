// 1. Give the struct a name based on the type (e.g., struct Maybe_int)
#define Maybe(T)                                                                                   \
    struct Maybe_##T {                                                                             \
        T value;                                                                                   \
        bool has_value;                                                                            \
    }

// 2. Use that same name in the constructors
// Note: We use the struct tag "struct Maybe_##T"
#define Some(T, val) ((struct Maybe_##T){.value = (val), .has_value = true})
#define None(T) ((struct Maybe_##T){.has_value = false})

// 3. Match remains the same
#define match(opt, name, some_block, none_block)                                                   \
    {                                                                                              \
        auto _opt = (opt);                                                                         \
        if (_opt.has_value) {                                                                      \
            typeof(_opt.value) name = _opt.value;                                                  \
            some_block                                                                             \
        } else {                                                                                   \
            none_block                                                                             \
        }                                                                                          \
    }

/* --- Example Usage ---


// Declare the specific Maybe type once
Maybe(int);

// Optional: create a clean typedef for it
typedef struct Maybe_int MaybeInt;

MaybeInt get_val() {
    printf("Evaluating...\n");
    return Some(int, 42); // This now works!
}

int main() {
    MaybeInt res = get_val();

    match(res, it, {
        printf("Value: %d\n", it);
    }, {
        printf("Nothing\n");
    });
}

*/
