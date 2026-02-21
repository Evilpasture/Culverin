#include "culverin_arg_indices.h"
#include "culverin_types.h"

void culverin_init_all_parsers(void) {
  // 1. Dummies for Type Deduction (Used by FP_ARG macros)
  PyObject *o;
  float f;
  double d;
  int i;
  uint32_t u;
  uint64_t k;
  bool b;

  // --- A. PHYSICS BODY CREATION ---
  {
    FastArgSpec specs[] = {[IDX_POS] = FP_ARG("pos", o),
                           [IDX_ROT] = FP_ARG("rot", o),
                           [IDX_SIZE] = FP_ARG("size", o),
                           [IDX_SHAPE] = FP_ARG("shape", i),
                           [IDX_MOTION] = FP_ARG("motion", i),
                           [IDX_USER_DATA] = FP_ARG("user_data", k),
                           [IDX_SENSOR] = FP_ARG("is_sensor", b),
                           [IDX_MASS] = FP_ARG("mass", f),
                           [IDX_CAT] = FP_ARG("category", u),
                           [IDX_MASK] = FP_ARG("mask", u),
                           [IDX_FRIC] = FP_ARG("friction", f),
                           [IDX_REST] = FP_ARG("restitution", f),
                           [IDX_MAT] = FP_ARG("material_id", u),
                           [IDX_CCD] = FP_ARG("ccd", b)};
    memcpy(BodySpecs, specs, sizeof(specs));
    FastParse_Init(&BodyParser, BodySpecs, IDX_BODY_COUNT);
  }

  // --- B. SET POSITION ---
  {
    FastArgSpec specs[] = {[IDX_SETPOS_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_SETPOS_X] = FP_REQ_ARG("x", d),
                           [IDX_SETPOS_Y] = FP_REQ_ARG("y", d),
                           [IDX_SETPOS_Z] = FP_REQ_ARG("z", d)};
    memcpy(SetPosSpecs, specs, sizeof(specs));
    FastParse_Init(&SetPosParser, SetPosSpecs, IDX_SETPOS_COUNT);
  }

  // --- C. SET VELOCITY ---
  {
    FastArgSpec specs[] = {[IDX_SETVEL_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_SETVEL_VX] = FP_REQ_ARG("vx", f),
                           [IDX_SETVEL_VY] = FP_REQ_ARG("vy", f),
                           [IDX_SETVEL_VZ] = FP_REQ_ARG("vz", f)};
    memcpy(SetVelSpecs, specs, sizeof(specs));
    FastParse_Init(&SetVelParser, SetVelSpecs, IDX_SETVEL_COUNT);
  }

  {
    PyObject *o;
    int i;
    FastArgSpec specs[] = {[IDX_BATCH_POSITIONS] = FP_REQ_ARG("positions", o),
                           [IDX_BATCH_SIZES] = FP_REQ_ARG("sizes", o),
                           [IDX_BATCH_SHAPE_TYPE] = FP_ARG("shape_type", i),
                           [IDX_BATCH_MOTION_TYPE] = FP_ARG("motion_type", i)};
    memcpy(BatchCreateSpecs, specs, sizeof(specs));
    FastParse_Init(&BatchCreateParser, BatchCreateSpecs, IDX_BATCH_COUNT);
  }
  {
    PyObject *o;
    FastArgSpec specs[] = {[IDX_BATCH_DESTROY_HANDLES] =
                               FP_REQ_ARG("handles", o)};
    memcpy(BatchDestroySpecs, specs, sizeof(specs));
    FastParse_Init(&BatchDestroyParser, BatchDestroySpecs,
                   IDX_BATCH_DESTROY_COUNT);
  }
  {
    uint64_t k;
    float f;
    FastArgSpec specs[] = {[IDX_SETROT_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_SETROT_X] = FP_REQ_ARG("x", f),
                           [IDX_SETROT_Y] = FP_REQ_ARG("y", f),
                           [IDX_SETROT_Z] = FP_REQ_ARG("z", f),
                           [IDX_SETROT_W] = FP_REQ_ARG("w", f)};
    memcpy(SetRotSpecs, specs, sizeof(specs));
    FastParse_Init(&SetRotParser, SetRotSpecs, IDX_SETROT_COUNT);
  }
  {
    uint64_t k;
    float f;
    FastArgSpec specs[] = {[IDX_SETLINVEL_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_SETLINVEL_X] = FP_REQ_ARG("x", f),
                           [IDX_SETLINVEL_Y] = FP_REQ_ARG("y", f),
                           [IDX_SETLINVEL_Z] = FP_REQ_ARG("z", f)};
    memcpy(SetLinVelSpecs, specs, sizeof(specs));
    FastParse_Init(&SetLinVelParser, SetLinVelSpecs, IDX_SETLINVEL_COUNT);
  }
  {
    uint64_t k;
    float f;
    FastArgSpec specs[] = {[IDX_SETANGVEL_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_SETANGVEL_X] = FP_REQ_ARG("x", f),
                           [IDX_SETANGVEL_Y] = FP_REQ_ARG("y", f),
                           [IDX_SETANGVEL_Z] = FP_REQ_ARG("z", f)};
    memcpy(SetAngVelSpecs, specs, sizeof(specs));
    FastParse_Init(&SetAngVelParser, SetAngVelSpecs, IDX_SETANGVEL_COUNT);
  }
  {
    uint64_t k;
    float f;
    FastArgSpec specs[] = {[IDX_IMPULSE_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_IMPULSE_X] = FP_REQ_ARG("x", f),
                           [IDX_IMPULSE_Y] = FP_REQ_ARG("y", f),
                           [IDX_IMPULSE_Z] = FP_REQ_ARG("z", f)};
    memcpy(ImpulseSpecs, specs, sizeof(specs));
    FastParse_Init(&ImpulseParser, ImpulseSpecs, IDX_IMPULSE_COUNT);
  }
  {
    uint64_t k;
    float f;
    JPH_Real r;
    FastArgSpec specs[] = {[IDX_IMPULSE_AT_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_IMPULSE_AT_IX] = FP_REQ_ARG("ix", f),
                           [IDX_IMPULSE_AT_IY] = FP_REQ_ARG("iy", f),
                           [IDX_IMPULSE_AT_IZ] = FP_REQ_ARG("iz", f),
                           [IDX_IMPULSE_AT_PX] = FP_REQ_ARG("px", r),
                           [IDX_IMPULSE_AT_PY] = FP_REQ_ARG("py", r),
                           [IDX_IMPULSE_AT_PZ] = FP_REQ_ARG("pz", r)};
    memcpy(ImpulseAtSpecs, specs, sizeof(specs));
    FastParse_Init(&ImpulseAtParser, ImpulseAtSpecs, IDX_IMPULSE_AT_COUNT);
  }
  {
    float f;
    FastArgSpec specs[] = {[IDX_GRAV_X] = FP_REQ_ARG("x", f),
                           [IDX_GRAV_Y] = FP_REQ_ARG("y", f),
                           [IDX_GRAV_Z] = FP_REQ_ARG("z", f)};
    memcpy(GravitySpecs, specs, sizeof(specs));
    FastParse_Init(&GravityParser, GravitySpecs, IDX_GRAV_COUNT);
  }
  {
    uint64_t k;
    double d;
    float f;
    PyObject *o;
    FastArgSpec specs[] = {[IDX_BUOY_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_BUOY_SURFACE_Y] = FP_REQ_ARG("surface_y", d),
                           [IDX_BUOY_BUOYANCY] = FP_ARG("buoyancy", f),
                           [IDX_BUOY_LIN_DRAG] = FP_ARG("linear_drag", f),
                           [IDX_BUOY_ANG_DRAG] = FP_ARG("angular_drag", f),
                           [IDX_BUOY_DT] = FP_ARG("dt", f),
                           [IDX_BUOY_VEL] = FP_ARG("fluid_velocity", o)};
    memcpy(BuoyancySpecs, specs, sizeof(specs));
    FastParse_Init(&BuoyancyParser, BuoyancySpecs, IDX_BUOY_COUNT);
  }
  {
    PyObject *o;
    JPH_Real r;
    float f;
    FastArgSpec specs[] = {[IDX_BBUOY_HANDLES] = FP_REQ_ARG("handles", o),
                           [IDX_BBUOY_SURFACE_Y] = FP_REQ_ARG("surface_y", r),
                           [IDX_BBUOY_BUOYANCY] = FP_ARG("buoyancy", f),
                           [IDX_BBUOY_LIN_DRAG] = FP_ARG("linear_drag", f),
                           [IDX_BBUOY_ANG_DRAG] = FP_ARG("angular_drag", f),
                           [IDX_BBUOY_DT] = FP_ARG("dt", f),
                           [IDX_BBUOY_VEL] = FP_ARG("fluid_velocity", o)};
    memcpy(BatchBuoyancySpecs, specs, sizeof(specs));
    FastParse_Init(&BatchBuoyancyParser, BatchBuoyancySpecs, IDX_BBUOY_COUNT);
  }
  {
    PyObject *o;
    int i;
    float f;
    uint64_t k;
    uint32_t u;
    bool b;
    FastArgSpec specs[] = {[IDX_HULL_POS] = FP_REQ_ARG("pos", o),
                           [IDX_HULL_ROT] = FP_REQ_ARG("rot", o),
                           [IDX_HULL_POINTS] = FP_REQ_ARG("points", o),
                           [IDX_HULL_MOTION] = FP_ARG("motion", i),
                           [IDX_HULL_MASS] = FP_ARG("mass", f),
                           [IDX_HULL_USER_DATA] = FP_ARG("user_data", k),
                           [IDX_HULL_CAT] = FP_ARG("category", u),
                           [IDX_HULL_MASK] = FP_ARG("mask", u),
                           [IDX_HULL_MAT_ID] = FP_ARG("material_id", u),
                           [IDX_HULL_FRIC] = FP_ARG("friction", f),
                           [IDX_HULL_REST] = FP_ARG("restitution", f),
                           [IDX_HULL_CCD] = FP_ARG("ccd", b)};
    memcpy(ConvexHullSpecs, specs, sizeof(specs));
    FastParse_Init(&ConvexHullParser, ConvexHullSpecs, IDX_HULL_COUNT);
  }
  {
    uint64_t k;
    float f;
    FastArgSpec specs[] = {[IDX_TORQUE_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_TORQUE_X] = FP_REQ_ARG("x", f),
                           [IDX_TORQUE_Y] = FP_REQ_ARG("y", f),
                           [IDX_TORQUE_Z] = FP_REQ_ARG("z", f)};
    memcpy(TorqueSpecs, specs, sizeof(specs));
    FastParse_Init(&TorqueParser, TorqueSpecs, IDX_TORQUE_COUNT);
  }
  {
    uint64_t k;
    float f;
    FastArgSpec specs[] = {[IDX_FORCE_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_FORCE_X] = FP_REQ_ARG("x", f),
                           [IDX_FORCE_Y] = FP_REQ_ARG("y", f),
                           [IDX_FORCE_Z] = FP_REQ_ARG("z", f)};
    memcpy(ForceSpecs, specs, sizeof(specs));
    FastParse_Init(&ForceParser, ForceSpecs, IDX_FORCE_COUNT);
  }
  {
    uint64_t k;
    float f;
    FastArgSpec specs[] = {[IDX_ANGIMP_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_ANGIMP_X] = FP_REQ_ARG("x", f),
                           [IDX_ANGIMP_Y] = FP_REQ_ARG("y", f),
                           [IDX_ANGIMP_Z] = FP_REQ_ARG("z", f)};
    memcpy(AngImpSpecs, specs, sizeof(specs));
    FastParse_Init(&AngImpParser, AngImpSpecs, IDX_ANGIMP_COUNT);
  }
  {
    PyObject *o;
    int i;
    float f;
    uint64_t k;
    uint32_t u;
    bool b;
    FastArgSpec specs[] = {[IDX_CMP_POS] = FP_REQ_ARG("pos", o),
                           [IDX_CMP_ROT] = FP_REQ_ARG("rot", o),
                           [IDX_CMP_PARTS] = FP_REQ_ARG("parts", o),
                           [IDX_CMP_MOTION] = FP_ARG("motion", i),
                           [IDX_CMP_MASS] = FP_ARG("mass", f),
                           [IDX_CMP_USER_DATA] = FP_ARG("user_data", k),
                           [IDX_CMP_SENSOR] = FP_ARG("is_sensor", b),
                           [IDX_CMP_CAT] = FP_ARG("category", u),
                           [IDX_CMP_MASK] = FP_ARG("mask", u),
                           [IDX_CMP_MAT_ID] = FP_ARG("material_id", u),
                           [IDX_CMP_FRIC] = FP_ARG("friction", f),
                           [IDX_CMP_REST] = FP_ARG("restitution", f),
                           [IDX_CMP_CCD] = FP_ARG("ccd", b)};
    memcpy(CompoundSpecs, specs, sizeof(specs));
    FastParse_Init(&CompoundParser, CompoundSpecs, IDX_CMP_COUNT);
  }
  {
    PyObject *o;
    uint64_t k;
    uint32_t u;
    FastArgSpec specs[] = {[IDX_MESH_POS] = FP_REQ_ARG("pos", o),
                           [IDX_MESH_ROT] = FP_REQ_ARG("rot", o),
                           [IDX_MESH_VERTS] = FP_REQ_ARG("vertices", o),
                           [IDX_MESH_INDICES] = FP_REQ_ARG("indices", o),
                           [IDX_MESH_USER_DATA] = FP_ARG("user_data", k),
                           [IDX_MESH_CAT] = FP_ARG("category", u),
                           [IDX_MESH_MASK] = FP_ARG("mask", u)};
    memcpy(MeshSpecs, specs, sizeof(specs));
    FastParse_Init(&MeshParser, MeshSpecs, IDX_MESH_COUNT);
  }
  {
    uint64_t k;
    FastArgSpec specs[] = {[IDX_DESTROY_HANDLE] = FP_REQ_ARG("handle", k)};
    memcpy(DestroySpecs, specs, sizeof(specs));
    FastParse_Init(&DestroyParser, DestroySpecs, IDX_DESTROY_COUNT);
  }
  {
    uint64_t k;
    PyObject *o;
    FastArgSpec specs[] = {[IDX_SETTRNS_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_SETTRNS_POS] = FP_REQ_ARG("pos", o),
                           [IDX_SETTRNS_ROT] = FP_REQ_ARG("rot", o)};
    memcpy(SetTransformSpecs, specs, sizeof(specs));
    FastParse_Init(&SetTransformParser, SetTransformSpecs, IDX_SETTRNS_COUNT);
  }
  {
    uint64_t k;
    bool b;
    FastArgSpec specs[] = {[IDX_CCD_HANDLE] = FP_REQ_ARG("handle", k),
                           [IDX_CCD_ENABLED] = FP_REQ_ARG("enabled", b)};
    memcpy(CCDSpecs, specs, sizeof(specs));
    FastParse_Init(&CCDParser, CCDSpecs, IDX_CCD_COUNT);
  }
}