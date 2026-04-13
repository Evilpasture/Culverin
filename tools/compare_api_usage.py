import os
import re
from pathlib import Path

# Configuration
JOLTC_HEADER = Path("extern/JoltC/include/joltc.h")
SOURCE_DIR = Path("src/culverin")
EXTENSIONS = {".c", ".h", ".cpp"}

def get_jolt_functions(header_path):
    """Extracts all JPH_CAPI function names from the header."""
    if not header_path.exists():
        print(f"Error: Header not found at {header_path}")
        return set()
    
    # Matches JPH_CAPI return_type JPH_FunctionName(args);
    pattern = re.compile(r"JPH_CAPI\s+[\w\*]+\s+(JPH_\w+)\s*\(")
    
    with open(header_path, "r", encoding="utf-8") as f:
        content = f.read()
        return set(pattern.findall(content))

def analyze_usage(source_dir, api_functions):
    """Searches for API function usage in the source directory."""
    used_functions = set()
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if Path(file).suffix in EXTENSIONS:
                path = Path(root) / file
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    for func in api_functions:
                        # Check for the function name followed by an opening parenthesis
                        if func in content:
                            used_functions.add(func)
                            
    return used_functions

def main():
    print(f"--- JoltC API Usage Analysis ---")
    
    # 1. Get total API surface
    api_functions = get_jolt_functions(JOLTC_HEADER)
    if not api_functions:
        return
    
    total_count = len(api_functions)
    print(f"Total functions defined in JoltC: {total_count}")

    # 2. Analyze source files
    used_functions = analyze_usage(SOURCE_DIR, api_functions)
    unused_functions = api_functions - used_functions
    
    used_count = len(used_functions)
    unused_count = len(unused_functions)
    usage_percent = (used_count / total_count) * 100

    # 3. Output Results
    print(f"Functions used in Culverin:    {used_count}")
    print(f"Functions unused:              {unused_count}")
    print(f"API Coverage:                  {usage_percent:.2f}%")
    
    print("\n--- Top Unused Categories (Samples) ---")
    # Group by prefix to see what subsystems are missing (e.g., JPH_WheeledVehicle)
    categories = {}
    for func in unused_functions:
        parts = func.split('_')
        cat = "_".join(parts[:2]) if len(parts) > 1 else "Other"
        categories[cat] = categories.get(cat, 0) + 1
    
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_cats[:10]:
        print(f"{cat:<30} : {count} unused functions")

    # Optional: Write unused list to file for review
    with open("unused_api.txt", "w") as f:
        f.write("\n".join(sorted(unused_functions)))
    print(f"\nFull list of unused functions written to unused_api.txt")

if __name__ == "__main__":
    main()