import re
from pathlib import Path
from dataclasses import dataclass

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
SOURCE_ROOT = SCRIPT_DIR.parent / "src" / "culverin"
SCHEMA_HEADER_NAME = "culverin_arg_indices.h"

TYPE_MAP = {
    "PyObject *": "Any", "PyObject*": "Any", 
    "float": "float", "double": "float", "JPH_Real": "float",
    "int": "int", "uint32_t": "int", "uint64_t": "int", "int64_t": "int",
    "BodyHandle": "int", "ConstraintHandle": "int", 
    "bool": "bool", "char *": "str", "const char *": "str",
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
    "PhysicsWorld.get_user_data": "int",
    "PhysicsWorld.get_motion_type": "int",
    "PhysicsWorld.activate": "handle: int",
    "PhysicsWorld.deactivate": "handle: int",
    "PhysicsWorld.destroy_body": "handle: int",
    "PhysicsWorld.destroy_constraint": "handle: int",
    "PhysicsWorld.set_collision_filter": "handle: int, category: int, mask: int",
    "RagdollSettings.stabilize": "None",
    "Skeleton.finalize": "None",
}

CLASS_METHOD_TO_SCHEMA = {
    "PhysicsWorld.create_body": "SCHEMA_BODY",
    "PhysicsWorld.apply_impulse": "SCHEMA_VEC3",
    "PhysicsWorld.apply_angular_impulse": "SCHEMA_VEC3",
    "PhysicsWorld.apply_force": "SCHEMA_VEC3",
    "PhysicsWorld.apply_torque": "SCHEMA_VEC3",
    "PhysicsWorld.set_linear_velocity": "SCHEMA_VEC3",
    "PhysicsWorld.set_angular_velocity": "SCHEMA_VEC3",
    "PhysicsWorld.set_position": "SCHEMA_SET_POS",
    "PhysicsWorld.set_rotation": "SCHEMA_SET_ROT",
    "PhysicsWorld.activate": "SCHEMA_HANDLE_ONLY",
    "PhysicsWorld.deactivate": "SCHEMA_HANDLE_ONLY",
    "PhysicsWorld.destroy_body": "SCHEMA_HANDLE_ONLY",
    "PhysicsWorld.get_user_data": "SCHEMA_HANDLE_ONLY",
    "PhysicsWorld.get_index": "SCHEMA_HANDLE_ONLY",
    "PhysicsWorld.is_alive": "SCHEMA_HANDLE_ONLY",
    "PhysicsWorld.destroy_constraint": "SCHEMA_HANDLE_ONLY",
    "PhysicsWorld.get_motion_type": "SCHEMA_HANDLE_ONLY",
    "PhysicsWorld.create_convex_hull": "SCHEMA_HC_HULL",
    "PhysicsWorld.create_compound_body": "SCHEMA_HC_COMP",
    "PhysicsWorld.create_mesh_body": "SCHEMA_MESH",
    "PhysicsWorld.set_transform": "SCHEMA_SET_TRNS",
    "PhysicsWorld.set_ccd": "SCHEMA_CCD",
    "PhysicsWorld.set_gravity": "SCHEMA_XYZ",
    "PhysicsWorld.raycast": "SCHEMA_RAYCAST",
    "PhysicsWorld.raycast_batch": "SCHEMA_RAYCAST_BATCH",
    "PhysicsWorld.shapecast": "SCHEMA_SHAPECAST",
    "PhysicsWorld.overlap_sphere": "SCHEMA_OVERLAP_SPHERE",
    "PhysicsWorld.overlap_aabb": "SCHEMA_OVERLAP_AABB",
    "PhysicsWorld.register_material": "SCHEMA_REG_MAT",
    "PhysicsWorld.set_constraint_target": "SCHEMA_SET_CONSTR_TARGET",
    "PhysicsWorld.step": "SCHEMA_STEP",
    "PhysicsWorld.create_bodies_batch": "SCHEMA_BATCH_CREATE",
    "PhysicsWorld.destroy_bodies_batch": "SCHEMA_BATCH_DESTROY",
    "PhysicsWorld.apply_buoyancy": "SCHEMA_BUOYANCY",
    "PhysicsWorld.apply_buoyancy_batch": "SCHEMA_BATCH_BUOYANCY",
    "PhysicsWorld.set_motion_type": "SCHEMA_SET_MOTION",
    "PhysicsWorld.set_collision_filter": "SCHEMA_COL_FILTER",
    "PhysicsWorld.create_constraint": "SCHEMA_CREATE_CONSTR",
    "PhysicsWorld.apply_impulse_at": "SCHEMA_IMPULSE_AT",
    "PhysicsWorld.set_user_data": "SCHEMA_SET_USER_DATA",
    "PhysicsWorld.create_heightfield": "SCHEMA_HEIGHTFIELD",
    "PhysicsWorld.load_state": "SCHEMA_LOAD_STATE",
    "PhysicsWorld.create_character": "SCHEMA_CREATE_CHAR",
    "PhysicsWorld.create_vehicle": "SCHEMA_CREATE_VEHICLE",
    "PhysicsWorld.create_tracked_vehicle": "SCHEMA_CREATE_TRACKED",
    "PhysicsWorld.create_ragdoll": "SCHEMA_CREATE_RAGDOLL",
    "PhysicsWorld.create_ragdoll_settings": "SCHEMA_RAGDOLL_SETTINGS",

    "Character.move": "SCHEMA_CHAR_MOVE",
    "Character.set_position": "SCHEMA_SET_POS_CHAR",
    "Character.set_rotation": "SCHEMA_SET_ROT_CHAR",
    "Character.set_strength": "SCHEMA_SET_STRENGTH_CHAR",

    "Vehicle.set_input": "SCHEMA_VEHICLE_INPUT",
    "Vehicle.set_tank_input": "SCHEMA_TANK_INPUT",
    "Vehicle.get_wheel_transform": "SCHEMA_WHEEL_IDX",
    "Vehicle.get_wheel_local_transform": "SCHEMA_WHEEL_IDX",

    "Skeleton.add_joint": "SCHEMA_ADD_JOINT",
    "Skeleton.get_joint_index": "SCHEMA_GET_JOINT_IDX",

    "RagdollSettings.add_part": "SCHEMA_RAGDOLL_ADD_PART",
    "Ragdoll.drive_to_pose": "SCHEMA_RAGDOLL_DRIVE"
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
        self.schemas = self._parse_schemas()

    def _improve_arg_type(self, arg_name: str, current_type: str) -> str:
        """Heuristics to upgrade 'Any' based on argument name."""
        if current_type != "Any": return current_type
        if arg_name in ("pos", "root_pos", "center", "dir", "direction", "scale"):
            return "tuple[float, float, float] | list[float]"
        if arg_name in ("rot", "root_rot"):
            return "tuple[float, float, float, float] | list[float]"
        if arg_name in ("size", "sizes"):
            return "tuple[float, ...] | list[float] | float"
        if arg_name in ("name", "drive"):
            return "str"
        if arg_name in ("handles", "vertices", "indices", "heights", "points", "matrices", "state"):
            return "bytes | bytearray | memoryview"
        return "Any"

    def _parse_schemas(self) -> dict[str, str]:
        header_path = next(self.root.rglob(SCHEMA_HEADER_NAME), None)
        if not header_path: return {}
        content = header_path.read_text()
        
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        blocks = re.findall(r'#define\s+(SCHEMA_\w+)\(X\)(.*?)(?=\n#define|\Z)', content, flags=re.DOTALL)
        
        parsed = {}
        for name, block in blocks:
            entries = re.findall(r'X\s*\(\s*[^,]+\s*,\s*"([^"]+)"\s*,\s*([^,]+)\s*,\s*(\d)\s*\)', block)
            py_args = []
            for arg_name, c_type, required in entries:
                t = TYPE_MAP.get(c_type.strip(), "Any")
                t = self._improve_arg_type(arg_name, t)
                
                if required == "1":
                    py_args.append(f"{arg_name}: {t}")
                else:
                    suffix = " | None = None" if "Any" not in t and "bytes" not in t else " = None"
                    py_args.append(f"{arg_name}: {t}{suffix}")
            parsed[name] = ", ".join(py_args)
        return parsed

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
            
            # Scrape Methods
            for class_name, block in method_pattern.findall(text):
                if class_name not in results: results[class_name] = []
                for m_name in re.findall(r'\{\s*"(\w+)"', block):
                    if m_name == "NULL": continue
                    
                    full_key = f"{class_name}.{m_name}"
                    info = MethodInfo(name=m_name)
                    
                    # 1. Base Args (Schema fallback to kwargs)
                    if full_key in CLASS_METHOD_TO_SCHEMA:
                        schema = CLASS_METHOD_TO_SCHEMA[full_key]
                        args = self.schemas.get(schema, "")
                        info.args = f"self, {args}" if args else "self"
                    else:
                        info.args = "self, *args, **kwargs"
                    
                    # 2. Argument Override
                    if full_key in ARG_HINTS:
                        hint = ARG_HINTS[full_key]
                        info.args = f"self, {hint}" if hint else "self"
                        
                    # 3. Return Type
                    info.return_type = RETURN_HINTS.get(full_key, self._infer_return_type(m_name))
                    
                    results[class_name].append(info)
                    
            # Scrape Properties
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
        
        for basic in ("Skeleton",):
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
    print(StubGenerator(SOURCE_ROOT).generate_stub())