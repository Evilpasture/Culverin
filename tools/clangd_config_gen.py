from pathlib import Path
import os

def generate_clangd():
    # Project location
    def get_project_root():
        current = Path(__file__).resolve()
        # Check current and every parent folder for the marker
        for parent in [current] + list(current.parents):
            if (parent / "CMakeLists.txt").exists():
                return parent
        return current.parent # Fallback

    project_root = get_project_root()
    print(f"Project root found at: {project_root}")
    
    # Specific freethreaded Python path
    python_base = Path("C:/Users/Admin/AppData/Local/Python/pythoncore-3.14t-64")
    python_include = python_base / "include"

    # Some Python installs nest headers deeper
    # e.g., include/python3.14t/Python.h
    final_python_path = python_include
    for root, dirs, files in os.walk(python_include):
        if "Python.h" in files:
            final_python_path = Path(root)
            break

    include_dirs = [
        f"{project_root.as_posix()}/extern/JoltC/include",
        f"{project_root.as_posix()}/extern/JoltPhysics",
        final_python_path.as_posix()
    ]

    # Verification
    print(f"Targeting Python Headers at: {final_python_path}")
    if not (final_python_path / "Python.h").exists():
        print("CRITICAL: Python.h still not found! Check the path manually.")

    # Format flags
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

    config = f"""# GENERATED FOR FREETHREADED PYTHON 3.14
CompileFlags:
  Add: [
      {formatted_flags}
  ]
  CompilationDatabase: "."

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
    print(".clangd file updated successfully.")

if __name__ == "__main__":
    generate_clangd()