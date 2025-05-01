#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>

struct ProteinAtom {
    int id;
    float x, y, z;
    float psi[8];
};


struct MoleculeAtom {
    int id;
    float x,y,z;
    float charge;
};


#endif




