import os
from pathlib import Path


def get_size_format(b, factor=1024, suffix="B") -> str | None:
    """Scale bytes to its proper format (e.g. 1.25KB)"""
    for unit in ["", "k", "M", "G", "T", "P"]:
        if b < factor:
            return f"{b:.1f} {unit}{suffix}"
        b /= factor


def generate_layout(root_dir=".") -> None:
    icons = {".py": "🐍", ".pyi": "📄", ".c": "⚙️ ", ".h": "📋", ".typed": "📄", "default": "📄"}

    print("Culverin Project Layout")

    # Sort files to match your structure: folders first, then alphabetically
    paths = sorted(Path(root_dir).rglob("*"), key=lambda p: (not p.is_dir(), p.name))

    for path in paths:
        # Skip hidden files/folders (like .git or .venv)
        if any(part.startswith(".") for part in path.parts):
            continue

        depth = len(path.relative_to(root_dir).parts) - 1
        spacer = "├── " if depth > 0 else ""
        indent = "│   " * (depth - 1) if depth > 1 else ""

        if path.is_file():
            ext = path.suffix
            icon = icons.get(ext, icons["default"])
            size = os.path.getsize(path)
            size_str = get_size_format(size)

            # Print in your exact format
            print(f"{indent}{spacer}{icon} {path.name} ({size_str})")
        elif path.is_dir():
            print(f"{indent}{spacer}📁 {path.name}/")


if __name__ == "__main__":
    generate_layout("src/culverin/")
