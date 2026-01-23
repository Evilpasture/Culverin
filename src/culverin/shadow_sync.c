#include "culverin.h"

void culverin_sync_shadow_buffers(PhysicsWorldObject* self) {
    JPH_BodyInterface* bi = self->body_interface;
    const size_t count = self->count;

    for (size_t i = 0; i < count; i++) {
        JPH_BodyID id = self->body_ids[i];
        if (id == JPH_INVALID_BODY_ID) continue;
        size_t idx = i * 4;

        // 1. Position (RVec3 - 32-byte alignment for AVX)
        JPH_STACK_ALLOC(JPH_RVec3, pos);
        JPH_BodyInterface_GetPosition(bi, id, pos);
        self->positions[idx + 0] = (float)pos->x;
        self->positions[idx + 1] = (float)pos->y;
        self->positions[idx + 2] = (float)pos->z;
        self->positions[idx + 3] = 0.0f;

        // 2. Rotation (Quat - 16-byte alignment)
        JPH_STACK_ALLOC(JPH_Quat, rot);
        JPH_BodyInterface_GetRotation(bi, id, rot);
        self->rotations[idx + 0] = rot->x;
        self->rotations[idx + 1] = rot->y;
        self->rotations[idx + 2] = rot->z;
        self->rotations[idx + 3] = rot->w;

        // 3. Linear Velocity (Vec3 - 16-byte alignment)
        JPH_STACK_ALLOC(JPH_Vec3, lin);
        JPH_BodyInterface_GetLinearVelocity(bi, id, lin);
        self->linear_velocities[idx + 0] = lin->x;
        self->linear_velocities[idx + 1] = lin->y;
        self->linear_velocities[idx + 2] = lin->z;
        self->linear_velocities[idx + 3] = 0.0f;

        // 4. Angular Velocity (Vec3 - 16-byte alignment)
        JPH_STACK_ALLOC(JPH_Vec3, ang);
        JPH_BodyInterface_GetAngularVelocity(bi, id, ang);
        self->angular_velocities[idx + 0] = ang->x;
        self->angular_velocities[idx + 1] = ang->y;
        self->angular_velocities[idx + 2] = ang->z;
        self->angular_velocities[idx + 3] = 0.0f;
    }
}