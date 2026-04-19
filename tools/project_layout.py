import os
from pathlib import Path

def get_size_format(b, factor=1024, suffix="B") -> str:
    for unit in ["", "k", "M", "G", "T", "P"]:
        if b < factor:
            return f"{b:.1f}{unit}{suffix}"
        b /= factor
    return f"{b:.1f}P{suffix}"

def count_lines(path: Path) -> int:
    try:
        if path.suffix in ('.c', '.h', '.cpp', '.py', '.pyi', '.cmake'):
            with open(path, 'rb') as f:
                return sum(1 for _ in f)
    except Exception:
        pass
    return 0

def analyze_directory(root_path: Path, max_depth: int = 3):
    icons = {
        ".py": "🐍", ".pyi": "📄", ".c": "⚙️ ", ".h": "📋", 
        ".cpp": "🛡️ ", ".pyd": "📦", ".pdb": "🔍", ".dll": "🔗",
        ".json": "📊", ".typed": "🏷️", "default": "📄"
    }

    stats = {"files": 0, "lines": 0, "size": 0}
    
    print(f"\n--- Analyzing: {root_path.name}/ ---")

    # Filter out hidden files and __pycache__
    paths = sorted(
        [p for p in root_path.rglob("*") if not any(part.startswith((".", "__")) for part in p.parts)],
        key=lambda p: (not p.is_dir(), p.name)
    )

    for path in paths:
        rel_path = path.relative_to(root_path)
        depth = len(rel_path.parts) - 1
        
        # Stop printing if too deep (but keep counting stats)
        if depth > max_depth:
            if path.is_file():
                stats["files"] += 1
                stats["size"] += os.path.getsize(path)
                stats["lines"] += count_lines(path)
            continue

        prefix = "    " * depth + ("└── " if depth > 0 else "")
        
        if path.is_dir():
            print(f"{prefix}📁 {path.name}/")
        else:
            ext = path.suffix
            icon = icons.get(ext, icons["default"])
            size = os.path.getsize(path)
            lines = count_lines(path)
            
            stats["files"] += 1
            stats["size"] += size
            stats["lines"] += lines
            
            size_str = get_size_format(size)
            line_str = f"{lines} lines" if lines else ""
            print(f"{prefix}{icon} {path.name:<35} ({size_str:<8} {line_str})")

    return stats

def run_full_analysis():
    targets = ["src/culverin", "tests", "extern"]
    grand_stats = {}

    print("CULVERIN PROJECT ECOSYSTEM")
    print("=" * 60)

    for t in targets:
        p = Path(t)
        if p.exists():
            # We set a lower depth for extern to avoid flooding the terminal
            depth = 1 if t == "extern" else 3
            grand_stats[t] = analyze_directory(p, max_depth=depth)
        else:
            print(f"\n[!] Target {t} not found. Skipping.")

    print("\n" + "=" * 60)
    print(f"{'Directory':<20} | {'Files':<8} | {'Lines':<10} | {'Total Size'}")
    print("-" * 60)
    
    for dir_name, s in grand_stats.items():
        size_str = get_size_format(s['size'])
        print(f"{dir_name:<20} | {s['files']:<8} | {s['lines']:<10} | {size_str}")

if __name__ == "__main__":
    run_full_analysis()