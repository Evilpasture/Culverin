import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- IMPORT YOUR CONFIG GEN ---
import clangd_config_gen
from scikit_build_core.build import build_wheel

# --- SMART PATHING ---
TOOLS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = TOOLS_DIR.parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
TARGET_DIR = PROJECT_ROOT / "src" / "culverin"

IS_CI = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")
cpu_count = os.cpu_count() or 4


def update_dev_tooling() -> None:
    """Regenerates .clangd and links compile_commands.json for IDE support."""
    print(">>> Updating development tooling (clangd/compile_commands)...")

    # 1. Regenerate .clangd
    try:
        clangd_config_gen.generate_clangd()
    except Exception as e:
        print(f"Warning: Failed to regenerate .clangd: {e}")

    # 2. Link compile_commands.json
    # scikit-build-core usually puts this in the root of the build folder
    source_cc = BUILD_DIR / "compile_commands.json"
    target_cc = PROJECT_ROOT / "compile_commands.json"

    if source_cc.exists():
        # Remove old link/file if it exists
        if target_cc.exists() or target_cc.is_symlink():
            target_cc.unlink()

        try:
            # Try to symlink (Best for dev, changes in build reflected instantly)
            os.symlink(source_cc, target_cc)
            print(f"Link created: {target_cc} -> {source_cc}")
        except OSError:
            # Fallback to copy (Windows without Dev Mode or different drives)
            shutil.copy2(source_cc, target_cc)
            print(f"File copied: {target_cc} (Symlink not supported)")
    else:
        print("Warning: compile_commands.json not found in build directory.")


def alert(success: bool = True) -> None:
    """Audio cues for headless building."""
    if IS_CI:
        return
    if success:
        print("\a")
        time.sleep(0.1)
        print("\a")
    else:
        print("\a")
        time.sleep(1.0)
        print("\a")


def install_package() -> None:
    """Uses uv to install the newly built wheel into the current environment."""
    print(">>> Installing package via uv...")

    # 1. Find the wheel we just built in the dist directory
    wheels = list(DIST_DIR.glob("*.whl"))
    if not wheels:
        print("Warning: No wheel found in dist/ to install.")
        return

    # Get the most recently created wheel
    latest_wheel = max(wheels, key=os.path.getmtime)

    # 2. Find uv
    uv_path = shutil.which("uv")
    if not uv_path:
        print("Warning: 'uv' not found in PATH. Skipping auto-install.")
        print(f"Manual install: pip install {latest_wheel}")
        return

    # 3. Run the install
    try:
        # --force-reinstall ensures it updates even if the version number hasn't changed
        subprocess.run(
            [uv_path, "pip", "install", str(latest_wheel), "--force-reinstall"], check=True
        )
        print(f"Successfully installed: {latest_wheel.name}")
    except subprocess.CalledProcessError as e:
        print(f"Error during uv install: {e}")


def build_extension() -> None:
    build_status = "INCOMPLETE (Crashed/Interrupted)"
    start_time = time.time()
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    print(f"--- CULVERIN ONE-CLICK BUILD (Python {sys.version.split()[0]}) ---")

    config: dict[str, str | list[str]] = {
        "cmake.define.CMAKE_BUILD_TYPE": "Release",
        "cmake.define.DOUBLE_PRECISION": "ON",
        "cmake.define.JPH_DOUBLE_PRECISION": "ON",
        "cmake.define.CMAKE_C_COMPILER": "clang",
        "cmake.define.CMAKE_CXX_COMPILER": "clang++",
        "cmake.define.ENABLE_SANITIZER": "OFF",
        "cmake.define.CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "build.tool-args": [f"-j{cpu_count}"],
        "build-dir": str(BUILD_DIR),
    }

    LOG_FILE = PROJECT_ROOT / "build_log.txt"

    def log_event(msg: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

    log_event("-" * 50)
    log_event("SESSION START")

    try:
        print(">>> Compiling and packaging...")
        build_wheel(str(DIST_DIR), config_settings=config)

        # DEPLOY BINARIES
        extension = ".pyd" if sys.platform == "win32" else ".so"
        # Find .pyd/.so AND .pdb files
        binary_files = [f for f in BUILD_DIR.glob(f"**/*{extension}") if "CMakeFiles" not in str(f)]
        symbol_files = [f for f in BUILD_DIR.glob("**/*.pdb") if "CMakeFiles" not in str(f)]

        if not binary_files:
            raise FileNotFoundError("Build finished but no binary found.")

        for pyd in binary_files:
            shutil.copy2(pyd, TARGET_DIR / pyd.name)

        # Copy the symbols (the .pdb map)
        for pdb in symbol_files:
            shutil.copy2(pdb, TARGET_DIR / pdb.name)
            print(f">>> Deployed symbols: {pdb.name}")

        # --- NEW STEPS ---
        update_dev_tooling()
        install_package()

        print(f"\n{GREEN}========================================{RESET}")
        print(f"{GREEN}BUILD & INSTALL SUCCESSFUL{RESET}")
        print(f"{GREEN}========================================{RESET}")
        build_status = "SUCCESS"
        alert(success=True)

    except Exception as e:
        print(f"\n{RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{RESET}")
        print(f"{RED}BUILD FAILED{RESET}")
        print(f"{RED}Error: {e}{RESET}")
        print(f"{RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{RESET}")
        build_status = f"FAILED: {e!s}"
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
        sys.exit(0)
    except Exception:
        sys.exit(1)
