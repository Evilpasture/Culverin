#pragma once
#include "physics/PhysicsParams.h"
#include "joltc.h"

#ifdef __cplusplus
extern "C" {
#endif

void CPH_ParamsInit(CPH_ConstraintParams *p);

JPH_Constraint *CPH_CreateFixed(const CPH_ConstraintParams *p, JPH_Body *b1, JPH_Body *b2);

JPH_Constraint *CPH_CreatePoint(const CPH_ConstraintParams *p, JPH_Body *b1, JPH_Body *b2);

JPH_Constraint *CPH_CreateHinge(const CPH_ConstraintParams *p, JPH_Body *b1, JPH_Body *b2);

JPH_Constraint *CPH_CreateSlider(const CPH_ConstraintParams *p, JPH_Body *b1, JPH_Body *b2);

JPH_Constraint *CPH_CreateCone(const CPH_ConstraintParams *p, JPH_Body *b1, JPH_Body *b2);

JPH_Constraint *CPH_CreateDistance(const CPH_ConstraintParams *p, JPH_Body *b1, JPH_Body *b2);

#ifdef __cplusplus
}
#endif
