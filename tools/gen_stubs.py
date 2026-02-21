import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
SOURCE_ROOT = SCRIPT_DIR.parent / "src" / "culverin"
SCHEMA_HEADER_NAME = "culverin_arg_indices.h"

TYPE_MAP = {
    "PyObject*": "Any", "PyObject *": "Any", 
    "float": "float", "double": "float", "JPH_Real": "float",
    "int": "int", "uint32_t": "int", "uint64_t": "int", "int64_t": "int",
    "BodyHandle": "int", "ConstraintHandle": "int", 
    "bool": "bool", "char*": "str", "const char*": "str"
}

# Explicit overrides: "ClassName.method_name": (args_string, return_type)
SIGNATURE_HINTS = {
    "PhysicsWorld.save_state": ("", "bytes"),
    "PhysicsWorld.get_active_indices": ("", "bytes"),
    "PhysicsWorld.get_contact_events_raw": ("", "memoryview"),
    "PhysicsWorld.get_contact_events": ("", "list[Any]"),
    "PhysicsWorld.get_contact_events_ex": ("", "list[dict[str, Any]]"),
    "PhysicsWorld.get_render_state": ("alpha: float", "bytes"),
    "PhysicsWorld.get_body_stats": ("handle: int", "tuple[tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float]]"),
    "Character.get_position": ("", "tuple[float, float, float]"),
    "Character.get_render_transform": ("alpha: float", "tuple[tuple[float, float, float], tuple[float, float, float, float]]"),
    "Character.is_grounded": ("", "bool"),
    "Vehicle.get_wheel_count": ("", "int"),
    "PhysicsWorld.activate": ("", "None"),
    "PhysicsWorld.deactivate": ("", "None"),
    "PhysicsWorld.get_user_data": ("handle: int", "int"),
    "PhysicsWorld.get_motion_type": ("handle: int", "int"),
    "PhysicsWorld.destroy_body": ("handle: int", "None"),
    "PhysicsWorld.destroy_constraint": ("handle: int", "None"),
    "PhysicsWorld.set_collision_filter": ("handle: int, category: int, mask: int", "None"),
}

SCHEMA_MAP = {
    "SCHEMA_BODY": "create_body",
    "SCHEMA_VEC3": ["apply_impulse", "apply_angular_impulse", "apply_force", "apply_torque", "set_linear_velocity", "set_angular_velocity", "move"],
    "SCHEMA_SET_POS": "set_position",
    "SCHEMA_SET_ROT": "set_rotation",
    "SCHEMA_HANDLE_ONLY": ["activate", "deactivate", "destroy_body", "get_user_data", "get_index", "is_alive", "destroy_constraint", "set_strength", "get_motion_type"],
    "SCHEMA_HC_HULL": "create_convex_hull",
    "SCHEMA_HC_COMP": "create_compound_body",
    "SCHEMA_MESH": "create_mesh_body",
    "SCHEMA_SET_TRNS": "set_transform",
    "SCHEMA_CCD": "set_ccd",
    "SCHEMA_XYZ": "set_gravity",
    "SCHEMA_RAYCAST": "raycast",
    "SCHEMA_RAYCAST_BATCH": "raycast_batch",
    "SCHEMA_SHAPECAST": "shapecast",
    "SCHEMA_OVERLAP_SPHERE": "overlap_sphere",
    "SCHEMA_OVERLAP_AABB": "overlap_aabb",
    "SCHEMA_REG_MAT": "register_material",
    "SCHEMA_SET_CONSTR_TARGET": "set_constraint_target",
    "SCHEMA_STEP": "step",
    "SCHEMA_BATCH_CREATE": "create_bodies_batch",
    "SCHEMA_BATCH_DESTROY": "destroy_bodies_batch",
    "SCHEMA_BUOYANCY": "apply_buoyancy",
    "SCHEMA_BATCH_BUOYANCY": "apply_buoyancy_batch",
    "SCHEMA_SET_MOTION": "set_motion_type",
    "SCHEMA_COL_FILTER": "set_collision_filter",
    "SCHEMA_CREATE_CONSTR": "create_constraint",
    "SCHEMA_IMPULSE_AT": "apply_impulse_at",
    "SCHEMA_SET_USER_DATA": "set_user_data",
}

@dataclass
class MethodInfo:
    name: str
    args: str = "self, *args, **kwargs"
    return_type: str = "Any"

class StubGenerator:
    def __init__(self, root: Path):
        self.root = root
        self.schemas = self._parse_schemas()
        self.method_to_schema = self._reverse_schema_map()

    def _parse_schemas(self) -> dict[str, str]:
        header_path = next(self.root.rglob(SCHEMA_HEADER_NAME), None)
        if not header_path: return {}
        content = header_path.read_text()
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        blocks = re.findall(r'#define\s+(SCHEMA_\w+)\(X\)\s*((?:.*\\\s*)*.*)', content)
        parsed = {}
        for name, block in blocks:
            entries = re.findall(r'X\s*\(\s*[^,]+\s*,\s*"([^"]+)"\s*,\s*([^,]+)\s*,\s*(\d)\s*\)', block)
            py_args = []
            for arg_name, c_type, required in entries:
                t = TYPE_MAP.get(c_type.strip(), "Any")
                if required == "1":
                    py_args.append(f"{arg_name}: {t}")
                else:
                    # If Any, don't use Optional syntax, just default to None
                    suffix = " | None = None" if t != "Any" else " = None"
                    py_args.append(f"{arg_name}: {t}{suffix}")
            parsed[name] = ", ".join(py_args)
        return parsed

    def _reverse_schema_map(self) -> dict[str, str]:
        rev = {}
        for schema, methods in SCHEMA_MAP.items():
            if isinstance(methods, list):
                for m in methods: rev[m] = schema
            else: rev[methods] = schema
        return rev

    def _infer_return_type(self, method_name: str) -> str:
        m = method_name.lower()
        if any(x in m for x in ("create", "get_index", "count")): return "int"
        if any(x in m for x in ("is_", "has_", "apply_buoyancy")): return "bool"
        if any(x in m for x in ("raycast", "shapecast", "overlap")): return "tuple[Any, ...] | None"
        if "get_render" in m: return "bytes"
        if any(m.startswith(x) for x in ("set_", "apply_", "activate", "deactivate", "destroy", "register_")) or m in ("step", "move"): return "None"
        return "Any"

    def get_class_methods(self) -> dict[str, list[MethodInfo]]:
        results = {}
        # Scrape PyMethodDef arrays
        pattern = re.compile(r'static\s+(?:const\s+)?PyMethodDef\s+(\w+)_methods\s*\[\]\s*=\s*\{(.*?)\}\s*;', re.DOTALL)

        for c_file in self.root.rglob("*.c"):
            for class_name, block in pattern.findall(c_file.read_text()):
                # Use the C class name exactly to maintain casing
                if class_name not in results: results[class_name] = []
                for m_name in re.findall(r'\{\s*"(\w+)"', block):
                    if m_name == "NULL": continue
                    full_key = f"{class_name}.{m_name}"
                    info = MethodInfo(name=m_name)
                    if full_key in SIGNATURE_HINTS:
                        args, ret = SIGNATURE_HINTS[full_key]
                        info.args = f"self, {args}" if args else "self"
                        info.return_type = ret
                    elif m_name in self.method_to_schema:
                        args = self.schemas.get(self.method_to_schema[m_name], "")
                        info.args = f"self, {args}" if args else "self"
                        info.return_type = self._infer_return_type(m_name)
                    else:
                        info.args = "self, *args, **kwargs"
                        info.return_type = self._infer_return_type(m_name)
                    results[class_name].append(info)
        return results

    def generate_stub(self) -> str:
        class_data = self.get_class_methods()
        lines = ["from __future__ import annotations", "from typing import Any", "", "'''Generated Stubs for Culverin Engine'''", ""]
        for cls in sorted(class_data.keys()):
            lines.append(f"class {cls}:")
            seen = set()
            for info in sorted(class_data[cls], key=lambda x: x.name):
                if info.name in seen: continue
                seen.add(info.name)
                lines.append(f"    def {info.name}({info.args}) -> {info.return_type}: ...")
            lines.append("")
        return "\n".join(lines)

if __name__ == "__main__":
    print(StubGenerator(SOURCE_ROOT).generate_stub())