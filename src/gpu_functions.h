#ifndef GPU_FUNCTIONS_H
#define GPU_FUNCTIONS_H

#include <vector>
#include <chrono>
#include "data_structures.h"

namespace gpu {
    // Funzioni di inizializzazione e cleanup
    void init_streams(int num_streams);
    void init_grid_texture(const std::vector<float>& cpu_grid_unique, int size);
    void cleanup_streams();
    void synchronize_all_streams();
    void init();
    void init_grid(const std::vector<float>& cpu_grid_unique, int grid_size);
    void cleanup();
    
    // Funzione di calcolo affinità per large dataset of molecules, asynchron
    std::vector<float> compute_affinity_AoS_async(const std::vector<MoleculeAtom>& molecule_atoms, int stream_id);
    std::vector<float> compute_affinity_channel_AoS_async(const std::vector<MoleculeAtom>& molecule_atoms, int stream_id);
   
    std::vector<float> compute_affinity_SoA_async(const MoleculeData& molecule_data, int stream_id);
   
    // texture mem + async
    std::vector<float> compute_affinity_texture_async(const std::vector<MoleculeAtom>& molecule_atoms, int stream_id);

    // Funzioni di calcolo dell'affinità - versione AoS
    std::vector<float> compute_affinity(const std::vector<MoleculeAtom>& molecule_atoms);
    std::vector<float> compute_affinity_channel(const std::vector<MoleculeAtom>& molecule_atoms);
    
    // Funzioni di calcolo dell'affinità - versione SoA
    std::vector<float> compute_affinity_soa(const MoleculeData& molecule_data);
    std::vector<float> compute_affinity_channel_soa(const MoleculeData& molecule_data);
    
    // Funzioni di benchmarking
    void evaluate_performance(const int type, const std::vector<MoleculeAtom>& molecule_atoms);
    void evaluate_performance_soa(const int type, const MoleculeData& molecule_data);

}

#endif