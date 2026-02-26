from pathlib import Path
import os
import sysconfig


def generate_clangd():
    # 1. Anchor to project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent

    # 2. Build Python Path (Absolute)
    include_path = Path(sysconfig.get_path("include"))
    final_python_path = include_path

    # Use _ for unused 'dirs' to satisfy the linter
    for root, _, files in os.walk(include_path):
        if "Python.h" in files:
            final_python_path = Path(root)
            break

    # 3. Define Dirs using project_root for portability
    # We convert to .as_posix() to ensure forward slashes on Windows
    include_dirs = [
        (project_root / "extern/JoltC/include").as_posix(),
        (project_root / "extern/JoltPhysics").as_posix(),
        final_python_path.as_posix(),
    ]

    flags = [f"-I{p}" for p in include_dirs]

    all_flags = [
        "-Wall",
        "-Wextra",
        "-m64",
        "-DJPH_DOUBLE_PRECISION",
        "-DPy_GIL_DISABLED=1",
        "-DMS_WIN64",
        "-D_CRT_SECURE_NO_WARNINGS",
        "-DMS_WINDOWS",
        "-fms-extensions",
    ] + flags

    formatted_flags = ",\n      ".join([f"'{f}'" for f in all_flags])

    # 4. Using rf"" (Raw f-string) to prevent escape sequence errors
    config = rf"""# GENERATED FOR FREETHREADED PYTHON 3.14
# This file is machine-generated.

CompileFlags:
  Add: [
      {formatted_flags}
  ]
  CompilationDatabase: "."

Index:
  Background: Skip

---
If:
  PathMatch: .*\.c
CompileFlags:
  Add: ["-std=c23"]

---
If:
  PathMatch: [.*\.cpp, .*\.hpp, .*\.cc, .*\.h]
CompileFlags:
  Add: ["-std=c++20", "-fno-exceptions", "-fno-rtti"]
"""

    with open(project_root / ".clangd", "w") as f:
        f.write(config)

    print(f"Success: .clangd updated at {project_root}")


if __name__ == "__main__":
    generate_clangd()

