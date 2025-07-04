#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#pragma once
#include <vector>
#include <array>

// Arrays of Struct (AoS)
struct ProteinAtom {
    int id;
    float x,y,z;
    float psi[8];
};

struct MoleculeAtom {
    int id;
    float x,y,z;
    float charge;
    std::vector<int> channel;
};

// Versione CUDA-compatibile di MoleculeAtom (AoS)
struct CudaMoleculeAtom {
    int id;
    float x, y, z;
    float charge;
    int channel[8]; // Array statico invece di std::vector
};

// Struct of Arrays (SoA) host
struct MoleculeData {
    std::vector<int> id;
    std::vector<float> x, y, z;
    std::vector<float> charge;
    std::array<std::vector<int>, 8> channel; 
    //                  column: atom
    //     row:channel   0 , 0 , 1 ,  ... for all atoms
    //                   1 , 0 , 0 ,  ... for all atoms         
};

// SoA gpu
struct MoleculeDataGPU {
    int* id;
    float* x;
    float* y;
    float* z;
    float* charge;
    int** channel;
};

struct GridConstants{
    int n_channel;
    float cell_size;
    int X, Y, Z;
    float x_min, y_min, z_min;
};

// Funzioni di conversione AoS
std::vector<CudaMoleculeAtom> convert_molecule_to_AoS_gpu(const std::vector<MoleculeAtom>& molecule_atoms);

// Funzioni di conversione SoA
MoleculeData convert_molecule_to_SoA(const std::vector<MoleculeAtom>& molecule_atoms);

GridConstants convert_to_grid_constants(const int n_channel, const float cell_size);

#endif