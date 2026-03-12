import os
import re
import json
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SOURCE_ROOT = PROJECT_ROOT / "src" / "culverin"

# We force the C code to execute inside SOURCE_ROOT, so it dumps here
SCHEMA_JSON_PATH = SOURCE_ROOT / "culverin_schema.json"
STUB_OUTPUT_PATH = SOURCE_ROOT / "_culverin_c.pyi"

# Exact C-Type to Python Type mapping
TYPE_MAP = {
    "float": "float", 
    "double": "float", 
    "JPH_Real": "float",
    "int": "int", 
    "uint32_t": "int", 
    "uint64_t": "int", 
    "int64_t": "int",
    "BodyHandle": "int", 
    "ConstraintHandle": "int", 
    "bool": "bool", 
    "char *": "str", 
    "const char *": "str",
    "PosStride": "tuple[float, float, float] | list[float]",
    "AuxStride": "tuple[float, float, float, float] | list[float]",
    "Vec3f": "tuple[float, float, float] | list[float]"
}

# Explicitly override ONLY the return types
RETURN_HINTS = {
    "PhysicsWorld.save_state": "bytes",
    "PhysicsWorld.get_active_indices": "bytes",
    "PhysicsWorld.get_contact_events_raw": "memoryview",
    "PhysicsWorld.get_contact_events": "list[tuple]",
    "PhysicsWorld.get_contact_events_ex": "list[dict[str, Any]]",
    "PhysicsWorld.get_render_state": "bytes",
    "PhysicsWorld.get_body_stats": "tuple[tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float]]",
    "PhysicsWorld.create_character": "'Character'",
    "PhysicsWorld.create_vehicle": "'Vehicle'",
    "PhysicsWorld.create_tracked_vehicle": "'Vehicle'",
    "PhysicsWorld.create_ragdoll_settings": "'RagdollSettings'",
    "PhysicsWorld.create_ragdoll": "'Ragdoll'",
    "Character.get_position": "tuple[float, float, float]",
    "Character.get_render_transform": "tuple[tuple[float, float, float], tuple[float, float, float, float]]",
    "Vehicle.get_wheel_transform": "tuple[tuple[float, float, float], tuple[float, float, float, float]]",
    "Vehicle.get_wheel_local_transform": "tuple[tuple[float, float, float], tuple[float, float, float, float]]",
    "Ragdoll.get_body_handles": "list[int]",
    "Ragdoll.get_debug_info": "list[dict[str, Any]]",
    "Skeleton.add_joint": "int",
    "Skeleton.get_joint_index": "int",
}

# Explicitly override ONLY the arguments (if not using schema)
ARG_HINTS = {
    "PhysicsWorld.get_render_state": "alpha: float",
    "Character.get_render_transform": "alpha: float",
    "Vehicle.get_wheel_transform": "index: int",
    "Vehicle.get_wheel_local_transform": "index: int",
    "PhysicsWorld.get_body_stats": "handle: int",
    "PhysicsWorld.get_user_data": "handle: int",
    "PhysicsWorld.get_motion_type": "handle: int",
    "PhysicsWorld.activate": "handle: int",
    "PhysicsWorld.deactivate": "handle: int",
    "PhysicsWorld.destroy_body": "handle: int",
    "PhysicsWorld.destroy_constraint": "handle: int",
    "PhysicsWorld.set_collision_filter": "handle: int, category: int, mask: int",
    "RagdollSettings.stabilize": "",  # Empty string means no extra args (just 'self')
    "Skeleton.finalize": "",          # Fixed syntax error
}

# Maps Python method names to the C `FastParser` registry name
CLASS_METHOD_TO_PARSER = {
    "PhysicsWorld.create_body": "Body",
    "PhysicsWorld.apply_impulse": "Impulse",
    "PhysicsWorld.apply_angular_impulse": "AngImpulse",
    "PhysicsWorld.apply_force": "Force",
    "PhysicsWorld.apply_torque": "Torque",
    "PhysicsWorld.set_linear_velocity": "SetLinVel",
    "PhysicsWorld.set_angular_velocity": "SetAngVel",
    "PhysicsWorld.set_position": "SetPos",
    "PhysicsWorld.set_rotation": "SetRot",
    "PhysicsWorld.activate": "Activate",
    "PhysicsWorld.deactivate": "Activate",
    "PhysicsWorld.destroy_body": "Destroy",
    "PhysicsWorld.get_user_data": "GetUserData",
    "PhysicsWorld.get_index": "Activate", # Reuses Activate parser (handle only)
    "PhysicsWorld.is_alive": "Activate",
    "PhysicsWorld.destroy_constraint": "DestroyConstr",
    "PhysicsWorld.get_motion_type": "GetMotion",
    "PhysicsWorld.create_convex_hull": "ConvexHull",
    "PhysicsWorld.create_compound_body": "Compound",
    "PhysicsWorld.create_mesh_body": "Mesh",
    "PhysicsWorld.set_transform": "SetTrns",
    "PhysicsWorld.set_ccd": "CCD",
    "PhysicsWorld.set_gravity": "Gravity",
    "PhysicsWorld.raycast": "Raycast",
    "PhysicsWorld.raycast_batch": "RayBatch",
    "PhysicsWorld.shapecast": "Shapecast",
    "PhysicsWorld.overlap_sphere": "OverlapSphere",
    "PhysicsWorld.overlap_aabb": "OverlapAABB",
    "PhysicsWorld.register_material": "RegMat",
    "PhysicsWorld.set_constraint_target": "SetConstrTarget",
    "PhysicsWorld.step": "Step",
    "PhysicsWorld.create_bodies_batch": "BatchCreate",
    "PhysicsWorld.destroy_bodies_batch": "BatchDestroy",
    "PhysicsWorld.apply_buoyancy": "Buoy",
    "PhysicsWorld.apply_buoyancy_batch": "BatchBuoy",
    "PhysicsWorld.set_motion_type": "SetMotion",
    "PhysicsWorld.set_collision_filter": "ColFilter",
    "PhysicsWorld.create_constraint": "CreateConstr",
    "PhysicsWorld.apply_impulse_at": "ImpulseAt",
    "PhysicsWorld.set_user_data": "SetUserData",
    "PhysicsWorld.create_heightfield": "Heightfield",
    "PhysicsWorld.load_state": "LoadState",
    "PhysicsWorld.create_character": "CreateChar",
    "PhysicsWorld.create_vehicle": "CreateVehicle",
    "PhysicsWorld.create_tracked_vehicle": "CreateTracked",
    "PhysicsWorld.create_ragdoll": "CreateRagdoll",
    "PhysicsWorld.create_ragdoll_settings": "RagdollSettings",

    "Character.move": "CharMove",
    "Character.set_position": "SetPosChar",
    "Character.set_rotation": "SetRotChar",
    "Character.set_strength": "SetStrengthChar",

    "Vehicle.set_input": "VehicleInput",
    "Vehicle.set_tank_input": "TankInput",
    "Vehicle.get_wheel_transform": "WheelIdx",
    "Vehicle.get_wheel_local_transform": "WheelIdx",

    "Skeleton.add_joint": "AddJoint",
    "Skeleton.get_joint_index": "GetJointIdx",

    "RagdollSettings.add_part": "RagdollAddPart",
    "Ragdoll.drive_to_pose": "RagdollDrive"
}

@dataclass
class MethodInfo:
    name: str
    args: str = "self, *args, **kwargs"
    return_type: str = "Any"
    is_property: bool = False

class StubGenerator:
    def __init__(self, root: Path):
        self.root = root
        self._generate_json_schema()
        self.schemas = self._load_schemas()

    def _generate_json_schema(self):
        """Runs the compiled C extension briefly to dump the internal JSON schema."""
        print(f"Generating C Schema JSON at {SOURCE_ROOT}...")
        
        env = os.environ.copy()
        # Ensure 'src' is in PYTHONPATH so we find the local culverin package
        src_dir = str(PROJECT_ROOT / "src")
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        
        script = "import culverin._culverin_c as c; c._dump_schema_json()"
        
        try:
            subprocess.run(
                [sys.executable, "-c", script], 
                check=True, 
                env=env,
                cwd=str(SOURCE_ROOT) # Forces C code to dump the file here
            )
        except subprocess.CalledProcessError as e:
            print(f"WARNING: C extension failed to dump schema. Return code: {e.returncode}")
            if not SCHEMA_JSON_PATH.exists():
                raise RuntimeError(f"No schema JSON found at {SCHEMA_JSON_PATH}, cannot proceed.")

    def _load_schemas(self) -> dict[str, str]:
        if not SCHEMA_JSON_PATH.exists():
            print(f"Warning: {SCHEMA_JSON_PATH} not found.")
            return {}
            
        with open(SCHEMA_JSON_PATH, "r") as f:
            raw_schemas = json.load(f)
            
        parsed = {}
        for parser_name, args_list in raw_schemas.items():
            py_args = []
            for arg in args_list:
                c_type = arg["type"].strip()
                t = TYPE_MAP.get(c_type, "Any")
                t = self._improve_arg_type(arg["name"], t)
                
                if arg["required"]:
                    py_args.append(f"{arg['name']}: {t}")
                else:
                    suffix = " | None = None" if "Any" not in t and "bytes" not in t else " = None"
                    py_args.append(f"{arg['name']}: {t}{suffix}")
            parsed[parser_name] = ", ".join(py_args)
            
        return parsed

    def _improve_arg_type(self, arg_name: str, current_type: str) -> str:
        """Heuristics to upgrade 'PyObject*' (Any) based on argument name."""
        if current_type != "Any": return current_type
        if arg_name in ("pos", "root_pos", "center", "dir", "direction", "scale", "velocity"):
            return "tuple[float, float, float] | list[float]"
        if arg_name in ("rot", "root_rot"):
            return "tuple[float, float, float, float] | list[float]"
        if arg_name in ("size", "sizes"):
            return "tuple[float, ...] | list[float] | float"
        if arg_name in ("name", "drive"):
            return "str"
        if arg_name in ("handles", "vertices", "indices", "heights", "points", "parts", "matrices", "state"):
            return "bytes | bytearray | memoryview | list"
        if arg_name in ("wheels", "tracks", "settings", "skeleton", "params", "motor"):
            return "list | dict | Any"
        return "Any"

    def _infer_return_type(self, method_name: str) -> str:
        m = method_name.lower()
        if any(x in m for x in ("create", "get_index", "count", "get_joint")): return "int"
        if any(x in m for x in ("is_", "has_", "apply_buoyancy")): return "bool"
        if any(x in m for x in ("raycast", "shapecast", "overlap")): return "tuple[Any, ...] | None"
        if any(m.startswith(x) for x in ("set_", "apply_", "activate", "deactivate", "destroy", "register_", "add_part")) or m in ("step", "move"): return "None"
        return "Any"

    def _infer_prop_type(self, prop_name: str) -> str:
        p = prop_name.lower()
        if p in ("positions", "rotations", "velocities", "angular_velocities", "user_data"): return "memoryview"
        if p in ("count", "shape_count", "max_bodies", "remaining_capacity", "handle", "wheel_count"): return "int"
        if p in ("time", "friction", "restitution"): return "float"
        if p.startswith("is_"): return "bool"
        return "Any"

    def get_class_methods(self) -> dict[str, list[MethodInfo]]:
        results = {}
        method_pattern = re.compile(r'static\s+(?:const\s+)?PyMethodDef\s+(\w+)_methods\s*\[\]\s*=\s*\{(.*?)\}\s*;', re.DOTALL)
        getset_pattern = re.compile(r'static\s+(?:const\s+)?PyGetSetDef\s+(\w+)_getset\s*\[\]\s*=\s*\{(.*?)\}\s*;', re.DOTALL)

        for c_file in self.root.rglob("*.c"):
            text = c_file.read_text()
            
            # 1. Scrape Methods
            for class_name, block in method_pattern.findall(text):
                if class_name not in results: results[class_name] = []
                for m_name in re.findall(r'\{\s*"(\w+)"', block):
                    if m_name == "NULL": continue
                    
                    full_key = f"{class_name}.{m_name}"
                    info = MethodInfo(name=m_name)
                    
                    # A. Map via JSON Schema
                    if full_key in CLASS_METHOD_TO_PARSER:
                        parser_name = CLASS_METHOD_TO_PARSER[full_key]
                        args_str = self.schemas.get(parser_name, "")
                        
                        if args_str:
                            info.args = f"self, {args_str}"
                        else:
                            # Fallback if schema lookup failed
                            info.args = "self, *args, **kwargs"
                    else:
                        info.args = "self, *args, **kwargs"
                    
                    # B. Manual Override (For simple functions without a parser)
                    if full_key in ARG_HINTS:
                        hint = ARG_HINTS[full_key]
                        if hint:
                            info.args = f"self, {hint}"
                        else:
                            info.args = "self" # Handles "" (no extra args)
                        
                    # C. Infer Return Type
                    info.return_type = RETURN_HINTS.get(full_key, self._infer_return_type(m_name))
                    
                    results[class_name].append(info)
                    
            # 2. Scrape Properties
            for class_name, block in getset_pattern.findall(text):
                if class_name not in results: results[class_name] = []
                for prop_name in re.findall(r'\{\s*"(\w+)"', block):
                    if prop_name == "NULL": continue
                    info = MethodInfo(name=prop_name, is_property=True)
                    info.return_type = self._infer_prop_type(prop_name)
                    results[class_name].append(info)
                    
        return results

    def generate_stub(self) -> str:
        class_data = self.get_class_methods()
        lines = [
            "from __future__ import annotations", 
            "from typing import Any", 
            "", 
            "'''Generated Stubs for Culverin Engine'''", 
            ""
        ]
        
        # Ensure base classes exist even if empty
        for basic in ("Skeleton", "Character", "Vehicle", "RagdollSettings", "Ragdoll"):
            if basic not in class_data:
                class_data[basic] = []
                
        for cls in sorted(class_data.keys()):
            lines.append(f"class {cls}:")
            seen = set()
            methods = sorted(class_data[cls], key=lambda x: (not x.is_property, x.name))
            
            if not methods:
                lines.append("    pass\n")
                continue
                
            for info in methods:
                if info.name in seen: continue
                seen.add(info.name)
                
                if info.is_property:
                    lines.append(f"    @property\n    def {info.name}(self) -> {info.return_type}: ...")
                else:
                    lines.append(f"    def {info.name}({info.args}) -> {info.return_type}: ...")
            lines.append("")
            
        return "\n".join(lines)

if __name__ == "__main__":
    generator = StubGenerator(SOURCE_ROOT)
    stub_content = generator.generate_stub()
    
    with open(STUB_OUTPUT_PATH, "w") as f:
        f.write(stub_content)
        
    print(f"Successfully generated stubs at: {STUB_OUTPUT_PATH.name}")