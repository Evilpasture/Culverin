#pragma once
#include "joltc.h"
#include "culverin_parsers.h"

void params_init(ConstraintParams *p);

JPH_Constraint *create_fixed(const ConstraintParams *p, JPH_Body *b1,
                                    JPH_Body *b2);

JPH_Constraint *create_point(const ConstraintParams *p, JPH_Body *b1,
                                    JPH_Body *b2);
                                    
JPH_Constraint *create_hinge(const ConstraintParams *p, JPH_Body *b1, JPH_Body *b2);

JPH_Constraint *create_slider(const ConstraintParams *p, JPH_Body *b1,
                                     JPH_Body *b2);

JPH_Constraint *create_cone(const ConstraintParams *p, JPH_Body *b1,
                                   JPH_Body *b2);

JPH_Constraint *create_distance(const ConstraintParams *p, JPH_Body *b1,
                                       JPH_Body *b2);