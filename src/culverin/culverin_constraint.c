#include "culverin_constraint.h"
#include "culverin_constraint_factory.h"

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyObject *PhysicsWorld_create_constraint(PhysicsWorldObject *self,
                                         PyObject *args, PyObject *kwds) {
  int type = 0;
  uint64_t h1 = 0;
  uint64_t h2 = 0;
  PyObject *params = NULL;
  PyObject *motor_dict = NULL; // NEW

  static char *kwlist[] = {"type",   "body1", "body2",
                           "params", "motor", NULL}; // Updated

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iKK|OO", kwlist, &type, &h1,
                                   &h2, &params, &motor_dict)) {
    return NULL;
  }

  // NEW: Explicitly forbid self-constraints (Jolt requires two distinct bodies)
  if (h1 == h2) {
    PyErr_SetString(PyExc_ValueError,
                    "Cannot create a constraint between a body and itself");
    return NULL;
  }

  ConstraintParams p;
  params_init(&p);
  int parse_ok = 1;
  switch (type) {
  case CONSTRAINT_FIXED:
    break;
  case CONSTRAINT_POINT:
    parse_ok = parse_point_params(params, &p);
    break;
  case CONSTRAINT_HINGE:
    parse_ok = parse_hinge_params(params, &p);
    break;
  case CONSTRAINT_SLIDER:
    parse_ok = parse_slider_params(params, &p);
    break;
  case CONSTRAINT_CONE:
    parse_ok = parse_cone_params(params, &p);
    break;
  case CONSTRAINT_DISTANCE:
    parse_ok = parse_distance_params(params, &p);
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "Unknown constraint type");
    return NULL;
  }
  if (!parse_ok) {
    return NULL;
  }

  if (motor_dict) {
    parse_motor_config(motor_dict, &p);
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t s1 = 0;
  uint32_t s2 = 0;
  if (!unpack_handle(self, h1, &s1) || self->slot_states[s1] != SLOT_ALIVE ||
      !unpack_handle(self, h2, &s2) || self->slot_states[s2] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid body handles");
    return NULL;
  }

  JPH_BodyID bid1 = self->body_ids[self->slot_to_dense[s1]];
  JPH_BodyID bid2 = self->body_ids[self->slot_to_dense[s2]];

  if (self->free_constraint_count == 0) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_MemoryError, "Max constraints reached");
    return NULL;
  }
  uint32_t c_slot = self->free_constraint_slots[--self->free_constraint_count];
  SHADOW_UNLOCK(&self->shadow_lock);

  const JPH_BodyLockInterface *lock_iface =
      JPH_PhysicsSystem_GetBodyLockInterface(self->system);
  JPH_BodyLockWrite lock1;
  JPH_BodyLockWrite lock2;

  // Sort for Deadlock Prevention
  if (bid1 < bid2) {
    JPH_BodyLockInterface_LockWrite(lock_iface, bid1, &lock1);
    JPH_BodyLockInterface_LockWrite(lock_iface, bid2, &lock2);
  } else {
    JPH_BodyLockInterface_LockWrite(lock_iface, bid2, &lock2);
    JPH_BodyLockInterface_LockWrite(lock_iface, bid1, &lock1);
  }

  JPH_Constraint *constraint = NULL;
  if (lock1.body && lock2.body) {
    // Resolve pointers (Safe because h1 != h2 guaranteed earlier)
    JPH_Body *b1 =
        (JPH_Body_GetID(lock1.body) == bid1) ? lock1.body : lock2.body;
    JPH_Body *b2 =
        (JPH_Body_GetID(lock1.body) == bid2) ? lock1.body : lock2.body;

    switch (type) {
    case CONSTRAINT_FIXED:
      constraint = create_fixed(&p, b1, b2);
      break;
    case CONSTRAINT_POINT:
      constraint = create_point(&p, b1, b2);
      break;
    case CONSTRAINT_HINGE:
      constraint = create_hinge(&p, b1, b2);
      break;
    case CONSTRAINT_SLIDER:
      constraint = create_slider(&p, b1, b2);
      break;
    case CONSTRAINT_CONE:
      constraint = create_cone(&p, b1, b2);
      break;
    case CONSTRAINT_DISTANCE:
      constraint = create_distance(&p, b1, b2);
      break;
    default:
      break; // Already handled above
    }
  }

  JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock1);
  JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock2);

  if (!constraint) {
    SHADOW_LOCK(&self->shadow_lock);
    self->free_constraint_slots[self->free_constraint_count++] = c_slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_RuntimeError,
                    "Jolt failed to create constraint instance");
    return NULL;
  }

  JPH_PhysicsSystem_AddConstraint(self->system, constraint);

  SHADOW_LOCK(&self->shadow_lock);
  self->constraints[c_slot] = constraint;
  self->constraint_states[c_slot] = SLOT_ALIVE;
  uint32_t gen = self->constraint_generations[c_slot];
  ConstraintHandle handle = ((uint64_t)gen << 32) | c_slot;
  SHADOW_UNLOCK(&self->shadow_lock);

  return PyLong_FromUnsignedLongLong(handle);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyObject *PhysicsWorld_destroy_constraint(PhysicsWorldObject *self,
                                          PyObject *args, PyObject *kwds) {
  uint64_t h = 0;
  static char *kwlist[] = {"handle", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &h)) {
    return NULL;
  }

  JPH_Constraint *c_to_destroy = NULL;

  // --- 1. RESOLUTION PHASE (Inside Shadow Lock) ---
  SHADOW_LOCK(&self->shadow_lock);

  // Guard against both Physics Step AND active Queries
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  auto slot = (uint32_t)(h & 0xFFFFFFFF);
  auto gen = (uint32_t)(h >> 32);

  // Validate identity
  if (slot >= self->constraint_capacity ||
      self->constraint_generations[slot] != gen ||
      self->constraint_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
    return NULL;
  }

  // Capture the pointer and IMMEDIATELY invalidate the slot
  c_to_destroy = self->constraints[slot];

  self->constraints[slot] = NULL;
  self->constraint_states[slot] = SLOT_EMPTY;
  self->constraint_generations[slot]++; // Increment generation to invalidate
                                        // stale handles
  self->free_constraint_slots[self->free_constraint_count++] = slot;

  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 2. JOLT DESTRUCTION PHASE (Outside Shadow Lock) ---
  // No Shadow-vs-Jolt deadlocks possible here!
  if (c_to_destroy) {
    // Automatic Body Wake-up
    // This is a "nice to have" - prevents objects from hanging in the air
    // when the joint holding them is deleted.
    if (JPH_Constraint_GetType(c_to_destroy) ==
        JPH_ConstraintType_TwoBodyConstraint) {
      auto *tbc = (JPH_TwoBodyConstraint *)c_to_destroy;
      JPH_Body *b1 = JPH_TwoBodyConstraint_GetBody1(tbc);
      JPH_Body *b2 = JPH_TwoBodyConstraint_GetBody2(tbc);

      // JPH_BodyInterface_ActivateBody is thread-safe
      if (b1) {
        JPH_BodyInterface_ActivateBody(self->body_interface,
                                       JPH_Body_GetID(b1));
      }
      if (b2) {
        JPH_BodyInterface_ActivateBody(self->body_interface,
                                       JPH_Body_GetID(b2));
      }
    }

    // Remove and Destroy
    JPH_PhysicsSystem_RemoveConstraint(self->system, c_to_destroy);
    JPH_Constraint_Destroy(c_to_destroy);
  }

  Py_RETURN_NONE;
}

PyObject *PhysicsWorld_set_constraint_target(PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds) {
  uint64_t h = 0;
  float target = 0.0f;
  static char *kwlist[] = {"handle", "target", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kf", kwlist, &h, &target)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, h, &slot) ||
      self->constraint_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid constraint handle");
    return NULL;
  }

  JPH_Constraint *c = self->constraints[slot];
  JPH_ConstraintType type = JPH_Constraint_GetType(c);
  JPH_ConstraintSubType sub = JPH_Constraint_GetSubType(c);

  // HINGE
  if (sub == JPH_ConstraintSubType_Hinge) {
    auto *hc = (JPH_HingeConstraint *)c;
    JPH_MotorState state = JPH_HingeConstraint_GetMotorState(hc);
    if (state == JPH_MotorState_Velocity) {
      JPH_HingeConstraint_SetTargetAngularVelocity(hc, target);
    } else if (state == JPH_MotorState_Position) {
      JPH_HingeConstraint_SetTargetAngle(hc, target);
    }
  }
  // SLIDER
  else if (sub == JPH_ConstraintSubType_Slider) {
    auto *sc = (JPH_SliderConstraint *)c;
    JPH_MotorState state = JPH_SliderConstraint_GetMotorState(sc);
    if (state == JPH_MotorState_Velocity) {
      JPH_SliderConstraint_SetTargetVelocity(sc, target);
    } else if (state == JPH_MotorState_Position) {
      JPH_SliderConstraint_SetTargetPosition(sc, target);
    }
  }

  // Wake up bodies!
  if (type == JPH_ConstraintType_TwoBodyConstraint) {
    JPH_Body *b1 = JPH_TwoBodyConstraint_GetBody1((JPH_TwoBodyConstraint *)c);
    JPH_Body *b2 = JPH_TwoBodyConstraint_GetBody2((JPH_TwoBodyConstraint *)c);
    JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b1));
    JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b2));
  }

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}