#pragma once
#include "joltc.h"

// Forward declarations
struct CharacterObject;
struct PhysicsWorldObject;

/* We expose the Procs table so the main module can assign it 
   when creating the Character Virtual instance.
*/
extern const JPH_CharacterContactListener_Procs char_listener_procs;