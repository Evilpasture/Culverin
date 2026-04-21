import subprocess
import argparse
import os
from pathlib import Path
from collections import Counter

def get_git_tracked_files(directory):
    """Fetch all files currently tracked by Git in the target directory."""
    try:
        # -z handles filenames with spaces or weird characters by using NUL delimiters
        result = subprocess.check_output(
            ["git", "-C", directory, "ls-files", "-z"],
            stderr=subprocess.STDOUT
        )
        return result.decode("utf-8").split('\0')[:-1]
    except subprocess.CalledProcessError:
        print(f"Error: '{directory}' does not appear to be a Git repository.")
        return []

def analyze_files(directory, file_list):
    """Analyze file extensions and line counts."""
    stats = Counter()
    total_lines = 0
    ext_lines = Counter()

    for file_path in file_list:
        full_path = Path(directory) / file_path
        if not full_path.is_file():
            continue

        ext = full_path.suffix or "[no extension]"
        stats[ext] += 1
        
        try:
            with open(full_path, "rb") as f:
                lines = sum(1 for _ in f)
                total_lines += lines
                ext_lines[ext] += lines
        except (PermissionError, OSError):
            continue

    return stats, total_lines, ext_lines

def main():
    parser = argparse.ArgumentParser(description="Analyze Git-tracked files in a directory.")
    parser.add_argument("dir", nargs="?", default=".", help="Directory to analyze (default: current)")
    args = parser.parse_args()

    target_dir = os.path.abspath(args.dir)
    print(f"--- Analyzing Git Repository: {target_dir} ---\n")

    files = get_git_tracked_files(target_dir)
    if not files:
        return

    ext_counts, total_lines, ext_lines = analyze_files(target_dir, files)

    # Output Results Table
    print(f"{'Extension':<15} | {'Count':<8} | {'Total Lines':<12} | {'% of Code'}")
    print("-" * 55)
    
    sorted_exts = sorted(ext_counts.items(), key=lambda x: ext_lines[x[0]], reverse=True)
    
    for ext, count in sorted_exts:
        lines = ext_lines[ext]
        percentage = (lines / total_lines * 100) if total_lines > 0 else 0
        print(f"{ext:<15} | {count:<8} | {lines:<12} | {percentage:>6.1f}%")

    print("-" * 55)
    print(f"Total Tracked Files: {len(files)}")
    print(f"Total Source Lines:  {total_lines}")

if __name__ == "__main__":
    main()