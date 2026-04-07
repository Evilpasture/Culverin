from pathlib import Path
import os
import sysconfig
import shutil
import subprocess

def get_macos_sdk_path():
    try:
        # Ask macOS where the SDK is located
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

    # 2. Identify the compiler path for Query-Driver
    # We want to tell clangd to ask the real compiler for system headers
    compiler_path = shutil.which("clang") or "/usr/bin/clang"

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
    ] + flags

    if sdk_path:
        all_flags.extend(["-isysroot", sdk_path])

    # Platform specific flags
    if os.name == 'nt':
        all_flags.extend(["-fms-compatibility", "-fms-extensions"])
    
    formatted_flags = ",\n      ".join([f"'{f}'" for f in all_flags])

    config = rf"""# GENERATED FOR FREETHREADED PYTHON 3.14
CompileFlags:
  Add: [
      {formatted_flags},
      "-ferror-limit=0"
  ]
  CompilationDatabase: "build"

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

    print(f"Success: .clangd updated with Query-Driver targeting {compiler_path}")

if __name__ == "__main__":
    generate_clangd()