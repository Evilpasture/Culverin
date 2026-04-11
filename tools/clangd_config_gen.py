from pathlib import Path
import os
import sysconfig
import shutil
import subprocess

def get_macos_sdk_path():
    try:
        return subprocess.check_output(["xcrun", "--show-sdk-path"]).decode("utf-8").strip()
    except:
        return None

def generate_clangd():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent if script_path.parent.name == "tools" else script_path.parent

    # 1. Find Python Include Path
    include_path = Path(sysconfig.get_path("include"))
    final_python_path = include_path
    for root, _, files in os.walk(include_path):
        if "Python.h" in files:
            final_python_path = Path(root)
            break

    include_dirs = [
        (project_root / "extern/JoltC/include").as_posix(),
        (project_root / "extern/JoltPhysics").as_posix(),
        final_python_path.as_posix(),
    ]

    flags = [f"-I{p}" for p in include_dirs]
    sdk_path = get_macos_sdk_path()

    all_flags = [
        "-Wall",
        "-Wextra",
        "-m64",
        "-DJPH_DOUBLE_PRECISION",
        "-DPy_GIL_DISABLED=1",
        
        # --- ATOMIC & CONCURRENCY SAFETY ---
        "-Watomic-implicit-seq-cst",   # CRITICAL: Warns when memory_order is not explicit
        "-Watomic-alignment",          # Warns if atomic ops will use a lock due to alignment
        "-Wthread-safety",             # Enables Clang's Thread Safety Analysis
        "-Wshadow",                    # Warns if locals shadow members (dangerous in C threads)
    ] + flags

    if sdk_path:
        all_flags.extend(["-isysroot", sdk_path])

    if os.name == 'nt':
        all_flags.extend(["-fms-compatibility", "-fms-extensions"])
    
    formatted_flags = ",\n      ".join([f"'{f}'" for f in all_flags])

    config = rf"""# GENERATED FOR CULVERIN ENGINE CONCURRENCY ANALYSIS
CompileFlags:
  Add: [
      {formatted_flags},
      "-ferror-limit=0"
  ]
  CompilationDatabase: "build"

Diagnostics:
  # This makes the implicit seq_cst warnings show up as errors in your editor
  # to ensure you never accidentally use the slowest memory barrier.
  UnusedIncludes: Strict
  CheckOptions:
    bugprone-assignment-in-if-condition: true
    bugprone-suspicious-include: true

---
# 1. THE C23 BLOCK
If:
  PathMatch: [.*\.c, .*/extern/JoltC/.*\.h]
CompileFlags:
  Add: [
      "-std=c23",
      "-Wno-c23-extensions",
      "-Wno-c2x-extensions"
  ]

---
# 2. THE C++23 BLOCK
If:
  PathMatch: [.*\.cpp, .*\.hpp, .*\.cc]
CompileFlags:
  Add: [
      "-std=c++23",
      "-frtti",
      "-fno-exceptions"
  ]
"""

    with open(project_root / ".clangd", "w") as f:
        f.write(config)

    print(f"Success: .clangd generated with Atomic Analysis flags.")

if __name__ == "__main__":
    generate_clangd()