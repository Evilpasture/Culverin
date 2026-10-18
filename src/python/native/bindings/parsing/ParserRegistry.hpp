#pragma once

#include "bindings/parsing/FastParser.hpp"

#include <cstddef>
#include <cstdint>
#include <new>

using ParseBodyHandle = uint64_t;
inline constexpr std::size_t kParserRegistrySize = 128;


struct CPH_WorldParsers {
    CPH::FastParser<PyObject *, PyObject *> WorldInit;
    CPH::FastParser<PyObject *, PyObject *, PyObject *, int, int, uint64_t, bool, float, uint32_t, uint32_t, float, float, uint32_t, bool, uint32_t> Body;
    CPH::FastParser<ParseBodyHandle, float, float, float> Impulse;
    CPH::FastParser<uint32_t> WheelIdx;
    CPH::FastParser<ParseBodyHandle, float, float, float> AngImpulse;
    CPH::FastParser<ParseBodyHandle, float, float, float> Force;
    CPH::FastParser<ParseBodyHandle, float, float, float> Torque;
    CPH::FastParser<ParseBodyHandle, float, float, float> SetLinVel;
    CPH::FastParser<ParseBodyHandle, float, float, float> SetAngVel;
    CPH::FastParser<ParseBodyHandle, float, float, float, JPH_Real, JPH_Real, JPH_Real> ImpulseAt;
    CPH::FastParser<ParseBodyHandle> HOnly;
    CPH::FastParser<ParseBodyHandle> Destroy;
    CPH::FastParser<ParseBodyHandle> Activate;
    CPH::FastParser<float, float, float> Gravity;
    CPH::FastParser<ParseBodyHandle, JPH_Real, JPH_Real, JPH_Real> SetPos;
    CPH::FastParser<ParseBodyHandle, double, float, float, float, float, PyObject *> Buoy;
    CPH::FastParser<PyObject *, double, float, float, float, float, PyObject *> BatchBuoy;
    CPH::FastParser<PyObject *, PyObject *, PyObject *, PyObject *, uint64_t, uint32_t, uint32_t> Mesh;
    CPH::FastParser<ParseBodyHandle, PyObject *, PyObject *> SetTrns;
    CPH::FastParser<ParseBodyHandle, bool> CCD;
    CPH::FastParser<PyObject *, PyObject *, PyObject *, int, float, uint64_t, bool, uint32_t, uint32_t, uint32_t, float, float, bool> ConvexHull;
    CPH::FastParser<PyObject *, PyObject *, PyObject *, int, float, uint64_t, bool, uint32_t, uint32_t, uint32_t, float, float, bool> Compound;
    CPH::FastParser<PyObject *, PyObject *, int, int> BatchCreate;
    CPH::FastParser<PyObject *> BatchDestroy;
    CPH::FastParser<ParseBodyHandle, float, float, float, float> SetRot;
    CPH::FastParser<float> Render;
    CPH::FastParser<PyObject *, PyObject *, float, ParseBodyHandle> Raycast;
    CPH::FastParser<PyObject *, PyObject *, float> RayBatch;
    CPH::FastParser<int, PyObject *, PyObject *, PyObject *, PyObject *, ParseBodyHandle> Shapecast;
    CPH::FastParser<PyObject *, float> OverlapSphere;
    CPH::FastParser<PyObject *, PyObject *> OverlapAABB;
    CPH::FastParser<ParseBodyHandle, uint64_t> SetUserData;
    CPH::FastParser<ParseBodyHandle> GetUserData;
    CPH::FastParser<ParseBodyHandle> GetMotion;
    CPH::FastParser<ParseBodyHandle, int> SetMotion;
    CPH::FastParser<ParseBodyHandle, uint32_t, uint32_t> ColFilter;
    CPH::FastParser<uint32_t, float, float> RegMat;
    CPH::FastParser<uint64_t, float> SetConstrTarget;
    CPH::FastParser<PyObject *, PyObject *, PyObject *, PyObject *, int, uint64_t, uint32_t, uint32_t, uint32_t, float, float> Heightfield;
    CPH::FastParser<bool, bool, bool, bool, bool> DebugData;
    CPH::FastParser<int, ParseBodyHandle, ParseBodyHandle, PyObject *, PyObject *> CreateConstr;
    CPH::FastParser<ParseBodyHandle> DestroyConstr;
    CPH::FastParser<float> Step;
    CPH::FastParser<CPH_PosStride, float, float, float, float> CreateChar;
    CPH::FastParser<PyObject *> LoadState;
    CPH::FastParser<uint64_t, PyObject *, PyObject *, PyObject *, PyObject *> CreateVehicle;
    CPH::FastParser<uint64_t, PyObject *, PyObject *, float, float> CreateTracked;
    CPH::FastParser<PyObject *, CPH_PosStride, CPH_AuxStride, uint64_t, uint32_t, uint32_t, uint32_t> CreateRagdoll;
    CPH::FastParser<PyObject *> RagdollSettings;
    CPH::FastParser<PyObject *, PyObject *, PyObject *, uint64_t, uint32_t, uint32_t, float, float, float, uint32_t, float, float, float, float, bool, bool, bool> CreateSoftBody;
    CPH::FastParser<ParseBodyHandle, uint32_t> GetSbVertex;
    CPH::FastParser<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> StressTest;
    CPH::FastParser<ParseBodyHandle, float, float, float, float, float, float, float> CreateShip;
    CPH_FastParser *registry[kParserRegistrySize]{};
    std::size_t registry_count{};
    void Init();
};

struct CPH_CharacterParsers {
    CPH::FastParser<CPH_Vec3f, float> CharMove;
    CPH::FastParser<CPH_PosStride> SetPosChar;
    CPH::FastParser<CPH_AuxStride> SetRotChar;
    CPH::FastParser<float> SetStrengthChar;
    CPH_FastParser *registry[kParserRegistrySize]{};
    std::size_t registry_count{};
    void Init();
};

struct CPH_VehicleParsers {
    CPH::FastParser<float, float, float, float> VehicleInput;
    CPH::FastParser<float, float, float> TankInput;
    CPH_FastParser *registry[kParserRegistrySize]{};
    std::size_t registry_count{};
    void Init();
};

struct CPH_ECSParsers {
    CPH::FastParser<uint64_t> RegEntityOnly;
    CPH::FastParser<uint32_t> RegCompOnly;
    CPH::FastParser<uint32_t> RegRegComp;
    CPH::FastParser<uint64_t, uint32_t, PyObject *> RegAdd;
    CPH::FastParser<uint64_t, uint32_t> RegEntComp;
    CPH::FastParser<PyObject *, uint32_t, int, int> RegSyncPhys;
    CPH_FastParser *registry[kParserRegistrySize]{};
    std::size_t registry_count{};
    void Init();
};

struct CPH_SkeletonParsers {
    CPH::FastParser<PyObject *, int> AddJoint;
    CPH::FastParser<PyObject *> GetJointIdx;
    CPH_FastParser *registry[kParserRegistrySize]{};
    std::size_t registry_count{};
    void Init();
};

struct CPH_RagdollParsers {
    CPH::FastParser<CPH_PosStride, CPH_AuxStride, PyObject *> RagdollDrive;
    CPH_FastParser *registry[kParserRegistrySize]{};
    std::size_t registry_count{};
    void Init();
};

struct CPH_RagdollSettingsParsers {
    CPH::FastParser<int, int, PyObject *, float, int, float, float, float, CPH_Vec3f, CPH_Vec3f, PyObject *> RagdollAddPart;
    CPH_FastParser *registry[kParserRegistrySize]{};
    std::size_t registry_count{};
    void Init();
};

struct CPH_ShipParsers {
    CPH::FastParser<float, float> ShipInput;
    CPH_FastParser *registry[kParserRegistrySize]{};
    std::size_t registry_count{};
    void Init();
};

struct CPH_SoftBodySharedSettingsParsers {
    CPH::FastParser<PyObject *, float, PyObject *> SbssAddVertex;
    CPH::FastParser<uint32_t, uint32_t, uint32_t, uint32_t> SbssAddFace;
    CPH::FastParser<PyObject *, PyObject *, PyObject *> SbssAddVertices;
    CPH::FastParser<PyObject *, PyObject *> SbssAddFaces;
    CPH::FastParser<float, int> SbssCreateConstraints;
    CPH_FastParser *registry[kParserRegistrySize]{};
    std::size_t registry_count{};
    void Init();
};

struct CPH_MathParsers {
    CPH::FastParser<float, float, float, float> MathPersp;
    CPH::FastParser<float, float, float, float, float, float> MathOrtho;
    CPH::FastParser<PyObject *, PyObject *, PyObject *> MathLookAt;
    CPH::FastParser<PyObject *, PyObject *, PyObject *> MathTRS;
    CPH::FastParser<PyObject *, PyObject *, PyObject *> MathTRSBatch;
    CPH::FastParser<PyObject *> MathMat;
    CPH::FastParser<PyObject *, PyObject *> MathMatPair;
    CPH::FastParser<PyObject *, PyObject *> MathMatVec;
    CPH::FastParser<PyObject *, PyObject *> MathMatBatch;
    CPH::FastParser<PyObject *, PyObject *, PyObject *> MathCull;
    CPH::FastParser<PyObject *, PyObject *> MathCullBatch;
    CPH::FastParser<PyObject *> MathVec3Batch;
    CPH::FastParser<float, float, float> MathEuler;
    CPH::FastParser<float, float, float, float> MathQuat;
    CPH::FastParser<PyObject *, PyObject *, float> MathSlerp;
    CPH::FastParser<PyObject *, PyObject *> MathQuatPair;
    CPH::FastParser<PyObject *, PyObject *, float> MathLerpBatch;
    CPH::FastParser<PyObject *, PyObject *> MathQuatVec;
    CPH::FastParser<PyObject *, PyObject *> MathQuatVecBatch;
    CPH::FastParser<PyObject *> MathQuatOp;
    CPH::FastParser<PyObject *, PyObject *, PyObject *> MathProject;
    CPH::FastParser<PyObject *, PyObject *, PyObject *> MathUnproject;
    CPH::FastParser<PyObject *, PyObject *> MathVecPair;
    CPH::FastParser<PyObject *, PyObject *, PyObject *, PyObject *> MathRayPlane;
    CPH::FastParser<PyObject *, float> MathAxisAngle;
    CPH::FastParser<PyObject *, PyObject *> MathDistBatch;
    CPH::FastParser<PyObject *> MathVecOp;
    CPH::FastParser<PyObject *, PyObject *> MathReflect;
    CPH::FastParser<PyObject *> MathEulerVec;
    CPH::FastParser<PyObject *> MathEulerBatch;
    CPH_FastParser *registry[kParserRegistrySize]{};
    std::size_t registry_count{};
    void Init();
};

struct CPH_ModuleParsers {
    CPH::FastParser<const char *> DumpSchema;
    CPH_FastParser *registry[kParserRegistrySize]{};
    std::size_t registry_count{};
    void Init();
};

namespace CPH {
template <typename ParserGroup>
[[nodiscard]] ParserGroup *NewParserGroup() noexcept {
    auto *group = new (std::nothrow) ParserGroup{};
    if (group == nullptr) {
        PyErr_NoMemory();
        return nullptr;
    }
    return group;
}
} // namespace CPH
