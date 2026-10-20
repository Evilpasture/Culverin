#include "physics/ConstraintFactory.hpp"

#include <Jolt/Physics/Constraints/ConeConstraint.h>
#include <Jolt/Physics/Constraints/DistanceConstraint.h>
#include <Jolt/Physics/Constraints/FixedConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/PointConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <cmath>

namespace CPH {
namespace {

constexpr float kCcfDefaultFrequency = 20.0f;
constexpr float kCcfDefaultDamping   = 1.0f;
constexpr float kCcfMaxTorque        = 1e6f;
constexpr float kCcfEpsilon          = 1e-6f;
constexpr float kCcfTinyEpsilon      = 1e-9f;

/// Normalizes (ax, ay, az), substituting a unit fallback when the axis is
/// near zero. `force_y` picks which component the fallback forces to 1; the
/// other components are left as given, matching the pre-native behaviour bit
/// for bit so recorded-scene replays stay identical.
JPH::Vec3 SanitizeAxis(float ax, float ay, float az, bool force_y) {
    if (ax * ax + ay * ay + az * az < kCcfEpsilon) {
        if (force_y) {
            ay = 1.0f;
        } else {
            ax = 1.0f;
        }
    }
    return JPH::Vec3(ax, ay, az).Normalized();
}

JPH::Constraint *CreateFixed(const ConstraintParams & /*unused*/, JPH::Body &body1, JPH::Body &body2) {
    JPH::FixedConstraintSettings s;
    s.mAutoDetectPoint = true;
    return s.Create(body1, body2);
}

JPH::Constraint *CreatePoint(const ConstraintParams &p, JPH::Body &body1, JPH::Body &body2) {
    JPH::PointConstraintSettings s;
    s.mPoint1 = JPH::RVec3(p.px, p.py, p.pz);
    s.mPoint2 = s.mPoint1;
    return s.Create(body1, body2);
}

JPH::Constraint *CreateHinge(const ConstraintParams &p, JPH::Body &body1, JPH::Body &body2) {
    JPH::HingeConstraintSettings s;
    s.mPoint1 = JPH::RVec3(p.px, p.py, p.pz);
    s.mPoint2 = s.mPoint1;

    const JPH::Vec3 axis = SanitizeAxis(p.ax, p.ay, p.az, /*force_y=*/true);

    // Compute normal from pivot-to-b2 projected onto the hinge plane,
    // so that angle=0 means "b2 at its initial position" and SetTargetAngle
    // is meaningful without the caller needing to know the reference frame.
    const JPH::RVec3 b2pos = body2.GetPosition();
    float px2 = (float)b2pos.GetX() - p.px;
    float py2 = (float)b2pos.GetY() - p.py;
    float pz2 = (float)b2pos.GetZ() - p.pz;

    const float dot = px2 * axis.GetX() + py2 * axis.GetY() + pz2 * axis.GetZ();
    px2 -= dot * axis.GetX();
    py2 -= dot * axis.GetY();
    pz2 -= dot * axis.GetZ();

    JPH::Vec3 norm;
    const float len = std::sqrt(px2 * px2 + py2 * py2 + pz2 * pz2);
    if (len > kCcfEpsilon) {
        const float inv = 1.0f / len;
        norm = JPH::Vec3(px2 * inv, py2 * inv, pz2 * inv);
    } else {
        // b2 is on the hinge axis itself — fall back to arbitrary perpendicular
        norm = axis.GetNormalizedPerpendicular();
    }

    s.mHingeAxis1  = axis;
    s.mHingeAxis2  = axis;
    s.mNormalAxis1 = norm;
    s.mNormalAxis2 = norm;
    s.mLimitsMin   = p.limit_min;
    s.mLimitsMax   = p.limit_max;

    s.mMotorSettings.mSpringSettings.mMode      = JPH::ESpringMode::FrequencyAndDamping;
    s.mMotorSettings.mSpringSettings.mFrequency = p.frequency > 0 ? p.frequency : kCcfDefaultFrequency;
    s.mMotorSettings.mSpringSettings.mDamping   = p.damping > 0 ? p.damping : kCcfDefaultDamping;
    s.mMotorSettings.mMaxTorqueLimit            = kCcfMaxTorque;
    s.mMotorSettings.mMinTorqueLimit            = -kCcfMaxTorque;

    JPH::HingeConstraint *c = static_cast<JPH::HingeConstraint *>(s.Create(body1, body2));

    if (p.has_motor && p.motor_type > 0) {
        c->SetMotorState(static_cast<JPH::EMotorState>(p.motor_type));
        if (p.motor_type == 1) {
            c->SetTargetAngularVelocity(p.motor_target);
        }
        if (p.motor_type == 2) {
            c->SetTargetAngle(p.motor_target);
        }
    }

    return c;
}

JPH::Constraint *CreateSlider(const ConstraintParams &p, JPH::Body &body1, JPH::Body &body2) {
    JPH::SliderConstraintSettings s;
    s.mPoint1 = JPH::RVec3(p.px, p.py, p.pz);
    s.mPoint2 = s.mPoint1;

    const JPH::Vec3 axis = SanitizeAxis(p.ax, p.ay, p.az, /*force_y=*/false);
    const JPH::Vec3 norm = axis.GetNormalizedPerpendicular();

    s.mSliderAxis1 = axis;
    s.mSliderAxis2 = axis;
    s.mNormalAxis1 = norm;
    s.mNormalAxis2 = norm;
    s.mLimitsMin   = p.limit_min;
    s.mLimitsMax   = p.limit_max;

    // --- Motor Support ---
    s.mMotorSettings.mSpringSettings.mMode      = JPH::ESpringMode::FrequencyAndDamping;
    s.mMotorSettings.mSpringSettings.mFrequency = p.frequency;
    s.mMotorSettings.mSpringSettings.mDamping   = p.damping;
    s.mMotorSettings.mMaxForceLimit             = p.max_force;
    s.mMotorSettings.mMinForceLimit             = -p.max_force;

    return s.Create(body1, body2);
}

JPH::Constraint *CreateCone(const ConstraintParams &p, JPH::Body &body1, JPH::Body &body2) {
    JPH::ConeConstraintSettings s;
    s.mPoint1 = JPH::RVec3(p.px, p.py, p.pz);
    s.mPoint2 = s.mPoint1;

    // SAFETY: If axis is zero, default to "UP" to prevent NaN explosion. The
    // cone uses a tighter threshold than the other constraint types.
    float ax = p.ax;
    float ay = p.ay;
    float az = p.az;
    if (ax * ax + ay * ay + az * az < kCcfTinyEpsilon) {
        ax = 0.0f;
        ay = 1.0f;
        az = 0.0f;
    }
    const JPH::Vec3 axis = JPH::Vec3(ax, ay, az).Normalized();

    s.mTwistAxis1    = axis;
    s.mTwistAxis2    = axis;
    s.mHalfConeAngle = p.half_cone_angle;

    return s.Create(body1, body2);
}

JPH::Constraint *CreateDistance(const ConstraintParams &p, JPH::Body &body1, JPH::Body &body2) {
    JPH::DistanceConstraintSettings s;

    // Check if the user provided a specific pivot point
    if (std::fabs(p.px) > kCcfEpsilon || std::fabs(p.py) > kCcfEpsilon || std::fabs(p.pz) > kCcfEpsilon) {
        s.mPoint1 = JPH::RVec3(p.px, p.py, p.pz);
        s.mPoint2 = s.mPoint1;
    } else {
        // Fallback: Default to current body centers if no pivot was provided
        s.mPoint1 = body1.GetPosition();
        s.mPoint2 = body2.GetPosition();
    }

    s.mMinDistance = p.limit_min;
    s.mMaxDistance = p.limit_max;

    return s.Create(body1, body2);
}

} // namespace

JPH::Constraint *CreateConstraint(ConstraintType type, const ConstraintParams &params,
                                  JPH::Body &body1, JPH::Body &body2) {
    switch (type) {
    case ConstraintType::Fixed:
        return CreateFixed(params, body1, body2);
    case ConstraintType::Point:
        return CreatePoint(params, body1, body2);
    case ConstraintType::Hinge:
        return CreateHinge(params, body1, body2);
    case ConstraintType::Slider:
        return CreateSlider(params, body1, body2);
    case ConstraintType::Cone:
        return CreateCone(params, body1, body2);
    case ConstraintType::Distance:
        return CreateDistance(params, body1, body2);
    default:
        return nullptr;
    }
}

} // namespace CPH
