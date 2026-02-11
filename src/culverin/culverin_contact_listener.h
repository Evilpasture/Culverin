#pragma once
#include "culverin.h"

extern const JPH_ContactListener_Procs contact_procs;

PyObject *PhysicsWorld_get_contact_events(PhysicsWorldObject *self,
                                          PyObject *args);

PyObject *PhysicsWorld_get_contact_events_ex(PhysicsWorldObject *self,
                                             PyObject *args);

PyObject *PhysicsWorld_get_contact_events_raw(PhysicsWorldObject *self,
                                              PyObject *args);