#!/usr/bin/env python3
import argparse
import os
import subprocess
import re
from collections import defaultdict
from pathlib import Path

# --- ANSI TERMINAL COLORS ---
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# --- ARCHITECTURE CATEGORIES ---
SYSTEMS_EXTS = {'.c', '.cpp', '.h', '.hpp', '.cc', '.cxx', '.inl'}
PYTHON_EXTS = {'.py', '.pyi', '.pyx', '.pyd'}
BUILD_EXTS = {'.cmake', '.txt', '.sh', '.bat', '.json', '.toml', '.yml'}

# --- REGEX HEURISTICS (The "Audit" Rules) ---
# We hate NULL and MSVC. We love constexpr, static inline, and atomics.
RE_NULL = re.compile(r'\bNULL\b')
RE_MSVC = re.compile(r'_MSC_VER')
RE_MACRO = re.compile(r'^\s*#\s*define\s+[A-Za-z0-9_]+')
RE_CONSTEXPR = re.compile(r'\bconstexpr\b')
RE_INLINE = re.compile(r'\bstatic\s+inline\b|\b[[gnu::always_inline]]\b')
RE_ATOMIC = re.compile(r'\batomic_|\bmemory_order_|\bstd::atomic\b')
RE_LOCK = re.compile(r'\bmutex|\bLOCK|\bunlock\b', re.IGNORECASE)

def get_git_tracked_files(directory):
    """Fetch tracked files, avoiding gitignore junk."""
    try:
        result = subprocess.check_output(
            ["git", "-C", directory, "ls-files", "-z"],
            stderr=subprocess.DEVNULL
        )
        return [f for f in result.decode("utf-8").split('\0') if f]
    except subprocess.CalledProcessError:
        print(f"{Colors.RED}Error: Not a git repository or git not installed.{Colors.RESET}")
        return []

def analyze_file(file_path, ext):
    """Stateful parsing for accurate metrics and C23 systems heuristics."""
    stats = {
        "code": 0, "comm": 0, "blank": 0,
        "null_count": 0, "msvc_count": 0, "macro_count": 0,
        "constexpr_count": 0, "inline_count": 0, "concurrency_score": 0
    }
    
    in_c_block = False
    in_py_block = False
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                raw_line = line.strip()
                
                # 1. Blank Lines
                if not raw_line:
                    stats["blank"] += 1
                    continue
                
                # 2. Stateful Comment Parsing
                is_comment = False
                if ext in SYSTEMS_EXTS:
                    if in_c_block:
                        is_comment = True
                        if '*/' in raw_line:
                            in_c_block = False
                    elif raw_line.startswith('//'):
                        is_comment = True
                    elif '/*' in raw_line:
                        is_comment = True
                        if '*/' not in raw_line:
                            in_c_block = True
                            
                elif ext in PYTHON_EXTS:
                    if in_py_block:
                        is_comment = True
                        if '"""' in raw_line or "'''" in raw_line:
                            in_py_block = False
                    elif raw_line.startswith('#'):
                        is_comment = True
                    elif raw_line.startswith('"""') or raw_line.startswith("'''"):
                        is_comment = True
                        if raw_line.count('"""') < 2 and raw_line.count("'''") < 2:
                            in_py_block = True

                if is_comment:
                    stats["comm"] += 1
                    continue
                
                # 3. If it's code, tally it up
                stats["code"] += 1
                
                # 4. Systems Quality Heuristics (Only for C/C++)
                if ext in SYSTEMS_EXTS:
                    if RE_NULL.search(line): stats["null_count"] += 1
                    if RE_MSVC.search(line): stats["msvc_count"] += 1
                    if RE_MACRO.match(line): stats["macro_count"] += 1
                    if RE_CONSTEXPR.search(line): stats["constexpr_count"] += 1
                    if RE_INLINE.search(line): stats["inline_count"] += 1
                    if RE_ATOMIC.search(line) or RE_LOCK.search(line): 
                        stats["concurrency_score"] += 1

    except Exception:
        pass # Ignore binary files or unreadable encodings
        
    return stats

def analyze_project(directory, file_list):
    file_data = []
    ext_stats = defaultdict(lambda: {"count": 0, "code": 0, "comm": 0, "blank": 0, "bytes": 0})
    sys_metrics = {"null": 0, "msvc": 0, "macro": 0, "constexpr": 0, "inline": 0, "concurrency": 0}
    category_counts = {"Systems (C/C++)": 0, "Scripting (Py)": 0, "Build/Config": 0, "Other": 0}
    
    for f_path in file_list:
        full_path = Path(directory) / f_path
        if not full_path.is_file(): 
            continue
        
        ext = full_path.suffix.lower() or "[no extension]"
        if full_path.name == "CMakeLists.txt": ext = ".cmake"
        
        # Categorize
        if ext in SYSTEMS_EXTS: category_counts["Systems (C/C++)"] += 1
        elif ext in PYTHON_EXTS: category_counts["Scripting (Py)"] += 1
        elif ext in BUILD_EXTS: category_counts["Build/Config"] += 1
        else: category_counts["Other"] += 1

        size = full_path.stat().st_size
        stats = analyze_file(full_path, ext)
        
        # Aggregate heuristics
        sys_metrics["null"] += stats["null_count"]
        sys_metrics["msvc"] += stats["msvc_count"]
        sys_metrics["macro"] += stats["macro_count"]
        sys_metrics["constexpr"] += stats["constexpr_count"]
        sys_metrics["inline"] += stats["inline_count"]
        sys_metrics["concurrency"] += stats["concurrency_score"]
        
        file_data.append({
            "path": f_path, "ext": ext, "size": size,
            "code": stats["code"], "comm": stats["comm"], "blank": stats["blank"]
        })
        
        # Aggregate extensions
        ext_stats[ext]["count"] += 1
        ext_stats[ext]["code"]  += stats["code"]
        ext_stats[ext]["comm"]  += stats["comm"]
        ext_stats[ext]["blank"] += stats["blank"]
        ext_stats[ext]["bytes"] += size

    return file_data, ext_stats, sys_metrics, category_counts

def print_bar(label, value, total, width=30, color=Colors.CYAN):
    percent = (value / total) if total > 0 else 0
    filled = int(percent * width)
    bar = ("█" * filled) + ("░" * (width - filled))
    print(f"{label:<15} | {color}{bar}{Colors.RESET} | {percent*100:>5.1f}% ({value:,})")

def main():
    parser = argparse.ArgumentParser(description="Strict Systems Codebase Audit")
    parser.add_argument("dir", nargs="?", default=".")
    args = parser.parse_args()

    target_dir = os.path.abspath(args.dir)
    files = get_git_tracked_files(target_dir)
    if not files: 
        return

    file_data, ext_stats, sys_metrics, cat_counts = analyze_project(target_dir, files)
    
    total_code = sum(s["code"] for s in ext_stats.values())
    total_comm = sum(s["comm"] for s in ext_stats.values())
    total_lines = total_code + total_comm + sum(s["blank"] for s in ext_stats.values())
    total_mb = sum(s['bytes'] for s in ext_stats.values()) / (1024 * 1024)

    print(f"\n{Colors.BOLD}{Colors.MAGENTA}=== PROJECT ARCHITECTURE AUDIT: {os.path.basename(target_dir)} ==={Colors.RESET}")
    print(f"Tracked Files : {len(files):,}")
    print(f"Total Lines   : {total_lines:,}")
    print(f"Total Code    : {total_code:,}")
    print(f"Total Size    : {total_mb:.2f} MB\n")

    # --- CATEGORY COMPOSITION ---
    print(f"{Colors.BOLD}--- CATEGORY COMPOSITION ---{Colors.RESET}")
    for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print_bar(cat, count, len(files), color=Colors.CYAN)
    print()

    # --- DETAILED EXTENSION BREAKDOWN ---
    print(f"{Colors.BOLD}--- FILE TYPE BREAKDOWN ---{Colors.RESET}")
    header = f"{'Extension':<12} | {'Files':<6} | {'Code':<10} | {'Docs/Comm':<10} | {'C/C Ratio'}"
    print(header)
    print("-" * len(header))

    sorted_exts = sorted(ext_stats.items(), key=lambda x: x[1]["code"], reverse=True)
    for ext, s in sorted_exts:
        if s["code"] == 0 and s["comm"] == 0: continue
        ratio = s["comm"] / s["code"] if s["code"] > 0 else 0
        c_color = Colors.GREEN if ratio > 0.2 else Colors.YELLOW
        print(f"{ext:<12} | {s['count']:<6} | {s['code']:<10,} | {s['comm']:<10,} | {c_color}{ratio:.2f}x{Colors.RESET}")
    print()

    # --- SYSTEMS CODE QUALITY METRICS ---
    print(f"{Colors.BOLD}--- C23 / SYSTEMS QUALITY AUDIT ---{Colors.RESET}")
    
    # Naughty list
    null_color = Colors.RED if sys_metrics["null"] > 0 else Colors.GREEN
    msvc_color = Colors.RED if sys_metrics["msvc"] > 0 else Colors.GREEN
    macro_color = Colors.YELLOW if sys_metrics["macro"] > sys_metrics["constexpr"] else Colors.GREEN
    
    print(f"Legacy '{null_color}NULL{Colors.RESET}' Usages      : {null_color}{sys_metrics['null']}{Colors.RESET}")
    print(f"MSVC Hacks (_MSC_VER)   : {msvc_color}{sys_metrics['msvc']}{Colors.RESET}")
    print(f"Legacy Macros (#define) : {macro_color}{sys_metrics['macro']}{Colors.RESET}")
    print("-" * 40)
    # Nice list
    print(f"Modern 'constexpr' uses : {Colors.GREEN}{sys_metrics['constexpr']}{Colors.RESET}")
    print(f"Static Inlines used     : {Colors.GREEN}{sys_metrics['inline']}{Colors.RESET}")
    print(f"Concurrency Footprint   : {Colors.CYAN}{sys_metrics['concurrency']} atomic/lock operations{Colors.RESET}\n")

    # --- TOP MONSTER FILES ---
    print(f"{Colors.BOLD}--- THE MONSTERS (TOP 10 LARGEST SOURCE FILES) ---{Colors.RESET}")
    sorted_files = sorted([f for f in file_data if f["code"] > 0], key=lambda x: x["code"], reverse=True)
    for i, f in enumerate(sorted_files[:10], 1):
        # Calculate localized comment ratio for the file to spot under-documented monoliths
        f_ratio = f["comm"] / f["code"] if f["code"] > 0 else 0
        warn = f"{Colors.RED}[!] Under-documented{Colors.RESET}" if f_ratio < 0.1 else ""
        print(f"{i:>2}. {f['code']:>6,} lines : {Colors.CYAN}{f['path']}{Colors.RESET} {warn}")
    print()

if __name__ == "__main__":
    main()