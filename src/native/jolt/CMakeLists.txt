# Jolt-specific runtime hooks are part of the native archive.
add_library(
    culverin_native_jolt
    OBJECT
        JoltHandler.cpp
        JoltHandler.h
        JoltConfig.h
        JoltUtils.h
)
target_link_libraries(
    culverin_native_jolt
    PRIVATE culverin_native_common culverin_native_cpp
)
culverin_configure_compile_target(culverin_native_jolt)

# This RTTI anchor is linked directly into the extension so its typeinfo
# references cannot be discarded from a static archive.
add_library(culverin_native_jolt_rtti OBJECT JoltRttiShim.cpp)
target_link_libraries(culverin_native_jolt_rtti PRIVATE culverin_native_common)
culverin_configure_compile_target(culverin_native_jolt_rtti)
if(MSVC)
    target_compile_options(culverin_native_jolt_rtti PRIVATE /GR)
else()
    target_compile_options(culverin_native_jolt_rtti PRIVATE -frtti)
endif()
