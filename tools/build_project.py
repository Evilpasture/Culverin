import sys
import shutil
import time
import os
from pathlib import Path
from scikit_build_core.build import build_wheel

# --- SMART PATHING ---
TOOLS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = TOOLS_DIR.parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
TARGET_DIR = PROJECT_ROOT / "src" / "culverin"

IS_CI = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")

def alert(success=True):
    """Audio cues for headless building."""
    if IS_CI: return  # Silence the bells in CI
    if success:
        print("\a") # Ding!
        time.sleep(0.1)
        print("\a") # Ding!
    else:
        print("\a") # Ding...
        time.sleep(1.0)
        print("\a") # ...Ding.

def build_extension():
    build_status = "INCOMPLETE (Crashed/Interrupted)"
    start_time = time.time()
    # ANSI Colors for QoL
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    print(f"--- CULVERIN ONE-CLICK BUILD (Python {sys.version.split()[0]}) ---")
    
    config = {
        "cmake.define.DOUBLE_PRECISION": "ON",
        "cmake.define.JPH_DOUBLE_PRECISION": "ON",
        "build-dir": str(BUILD_DIR),
    }

    LOG_FILE = PROJECT_ROOT / "build_log.txt"

    def log_event(msg):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

    log_event("-" * 50)
    log_event("SESSION START")

    try:
        print(">>> Compiling and packaging...")
        build_wheel(str(DIST_DIR), config_settings=config)

        extension = ".pyd" if sys.platform == "win32" else ".so"
        print(f">>> Deploying {extension} to {TARGET_DIR}...")
        # The "Platform-Aware" Glob
        binary_files = [f for f in BUILD_DIR.glob(f"**/*{extension}") if "CMakeFiles" not in str(f)]
        
        if not binary_files:
            raise FileNotFoundError("Build finished but no .pyd found.")

        for pyd in binary_files:
            shutil.copy2(pyd, TARGET_DIR / pyd.name)
        
        print(f"\n{GREEN}========================================{RESET}")
        print(f"{GREEN}BUILD SUCCESSFUL AND DEPLOYED{RESET}")
        print(f"{GREEN}========================================{RESET}")
        build_status = "SUCCESS"
        alert(success=True)

    except Exception as e:
        print(f"\n{RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{RESET}")
        print(f"{RED}BUILD FAILED{RESET}")
        print(f"{RED}Error: {e}{RESET}")
        print(f"{RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{RESET}")
        build_status = f"FAILED: {str(e)}"
        alert(success=False)
        raise
    finally:
        duration = round(time.time() - start_time, 2)
        log_event(f"RESULT: {build_status}")
        log_event(f"DURATION: {duration} seconds")
        log_event("SESSION END")
        log_event("-" * 50)

if __name__ == "__main__":
    try:
        build_extension()
        sys.exit(0) # The Green Checkmark equivalent
    except Exception:
        sys.exit(1) # The Red X equivalent