from pathlib import Path
import os
import sysconfig

def generate_clangd():
    # 1. Find the project root relative to this script
    script_path = Path(__file__).resolve()
    project_root = script_path.parent 
    # (Since this script likely lives in your root)

    # 2. Build Python Path (Keep this Absolute)
    include_path = Path(sysconfig.get_path("include"))
    final_python_path = include_path
    for root, dirs, files in os.walk(include_path):
        if "Python.h" in files:
            final_python_path = Path(root)
            break

    # 3. Define Dirs: Use Relative for local, Absolute for system
    include_dirs = [
        "extern/JoltC/include", # Relative to .clangd location
        "extern/JoltPhysics",   # Relative
        final_python_path.as_posix() # Absolute (System)
    ]

    flags = [f"-I{p}" for p in include_dirs]
    
    all_flags = [
        "-Wall", "-Wextra", "-m64",
        "-DJPH_DOUBLE_PRECISION",
        "-DPy_GIL_DISABLED=1",
        "-DMS_WIN64",
        "-D_CRT_SECURE_NO_WARNINGS",
        "-DMS_WINDOWS"
    ] + flags

    formatted_flags = ",\n      ".join([f"'{f}'" for f in all_flags])

    # 4. The Config with the "Linter Leash"
    config = f"""# GENERATED FOR FREETHREADED PYTHON 3.14
# This file is machine-generated. Do not edit manually.

CompileFlags:
  Add: [
      {formatted_flags}
  ]
  CompilationDatabase: "."

# Disable background indexing of the massive Jolt library
Index:
  Background: Skip

---
If:
  PathMatch: [.*\\.c, .*\\.h]
CompileFlags:
  Add: ["-std=c23", "-xc"]

---
If:
  PathMatch: [.*\\.cpp, .*\\.hpp, .*\\.cc]
CompileFlags:
  Add: ["-std=c++17", "-xc++", "-fno-exceptions", "-fno-rtti"]
"""

    with open(".clangd", "w") as f:
        f.write(config)
    print(f"Success: .clangd updated with {len(include_dirs)} include paths.")

if __name__ == "__main__":
    generate_clangd()