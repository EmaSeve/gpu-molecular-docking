#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>

#include <chrono>
#include <random>

#include <future>
#include <thread>

#include "data_structures.h"
#include "parser.h"
#include "gpu_functions.h"
#include "constants.h"

int X,Y,Z;

int grid_size;
std::vector<float> grid_unique;

float x_min = std::numeric_limits<float>::infinity();
float y_min = std::numeric_limits<float>::infinity();
float z_min = std::numeric_limits<float>::infinity();

float x_max = - std::numeric_limits<float>::infinity();
float y_max = - std::numeric_limits<float>::infinity();
float z_max = - std::numeric_limits<float>::infinity();


void compute_grid_dimension(const std::vector<ProteinAtom>& atoms){
    
    for(const auto & atom : atoms){
        if(atom.x > x_max) x_max = atom.x;
        if(atom.x < x_min) x_min = atom.x;

        if(atom.y > y_max) y_max = atom.y;
        if(atom.y < y_min) y_min = atom.y;

        if(atom.z > z_max) z_max = atom.z;
        if(atom.z < z_min) z_min = atom.z;
    }

    // traslated to the corner of the cube
    x_max += cell_size * 0.5;
    x_min -= cell_size * 0.5;

    y_max += cell_size * 0.5;
    y_min -= cell_size * 0.5;

    z_max += cell_size * 0.5;
    z_min -= cell_size * 0.5;


    X = static_cast<int>((x_max - x_min) / cell_size);
    Y = static_cast<int>((y_max - y_min) / cell_size);
    Z = static_cast<int>((z_max - z_min) / cell_size);

    grid_size = X * Y * Z; 
}

int compute_cell_index(const float& atom_x, const float& atom_y, const float& atom_z){
    
    // compute the relative cell in the grid
    int cell_x = static_cast<int>((atom_x - x_min) / cell_size);
    int cell_y = static_cast<int>((atom_y - y_min) / cell_size);
    int cell_z = static_cast<int>((atom_z - z_min) / cell_size);

    // index of the cell
    int index = cell_x + cell_y * (X) + cell_z * (X * Y);

    return index;
}

void init_grid(std::vector<ProteinAtom>& protein_atoms){

    compute_grid_dimension(protein_atoms);

    // dimension inizialized
    grid_unique.resize(grid_size * n_channel);

    // initialize the grid
    for(int i = 0; i < grid_size * n_channel; i++){
       grid_unique[i] = 0.0;
    }

    // fill each cell occupied by the protein with the 8 values of the channels
    for(auto &atom : protein_atoms){
        int cell_index = compute_cell_index(atom.x,atom.y,atom.z);
        int global_index = cell_index * n_channel;

        for(int i = 0; i < n_channel; i++){
            grid_unique[global_index + i] = atom.psi[i];
        }
    }
}

std::vector<float> compute_affinity(const std::vector<MoleculeAtom> & molecule_atoms){
    int M_atom_index = 0;
    std::vector<float> result(molecule_atoms.size() * n_channel);

    for(const auto& atom : molecule_atoms){
        int cell_index = compute_cell_index(atom.x,atom.y,atom.z);
        int global_index = cell_index * n_channel;

        for(int i = 0; i < n_channel; i++){
            result[M_atom_index * n_channel + i] = grid_unique[global_index + i] * atom.charge;
        }
        M_atom_index++;   
    }

    return result;
}

std::vector<float> compute_affinity_channel(const std::vector<MoleculeAtom> & molecule_atoms){
    int M_atom_index = 0;
    std::vector<float> result(molecule_atoms.size());

    for(const auto& atom : molecule_atoms){
        int cell_index = compute_cell_index(atom.x,atom.y,atom.z);
        int global_index = cell_index * n_channel;

        for(int i = 0; i < n_channel; i++){
            if(atom.channel[i] == 1)
                result[M_atom_index] = grid_unique[global_index + i] * atom.charge;
        }
        M_atom_index++;   
    }

    return result;
}

void print_result(const std::vector<MoleculeAtom> &molecule_atoms, const std::vector<float> &result){
    for(int i = 0; i<molecule_atoms.size();i++){
        int id = i + 1;
        int cell_index = compute_cell_index(molecule_atoms[i].x,molecule_atoms[i].y,molecule_atoms[i].z);
        int size = result.size() / molecule_atoms.size();
        std::cout<<"Atom id: "<<id<<", grid cell: "<<cell_index<<std::endl;

        for(int j = 0; j < size; j++){
                std::cout<<"  output on channel "<<j+1<<": "<<result[i * size + j]<<std::endl;
        }
    }
}

void valutate_performance_cpu(const int type, const std::vector<MoleculeAtom> &molecule_atoms) {
    const int num_runs = 100;
    std::chrono::duration<double> total_time(0);
    std::vector<float> result;
    
    for(int i = 0; i < num_runs; i++){
        auto start = std::chrono::high_resolution_clock::now();
        if(type == 1) {
            result = compute_affinity(molecule_atoms);
        } else if(type == 2) {
            result = compute_affinity_channel(molecule_atoms);
        }
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
    }
    
    double average_time = total_time.count() / num_runs;
    std::cout << "Average time used to compute " << 
              (type == 1 ? "general" : "channel") << 
              " affinity: " << average_time << std::endl;
}


std::vector<float> dataset_gpu_processing(const std::vector<MoleculeAtom>& large_dataset, const int type){
    
    const int NUM_STREAMS = 2;
    const size_t BATCH_SIZE = dataset_size / NUM_STREAMS;
    const int different_approach = 4;

    std::vector<float> final_results;

    std::cout << "\n===== DATASET GPU PROCESSING =====\n";
    std::cout << "Using: Batch processing + Async transfers \n";
    std::cout << "Batch size: " << BATCH_SIZE << ", Streams: " << NUM_STREAMS << std::endl;

       if(type == 0){
        std::cout << "\n --- general_affinity AoS ---\n";

        gpu::init_streams(NUM_STREAMS);

        std::vector<std::vector<MoleculeAtom>> batches;
        std::vector<std::vector<float>> batch_results;

        for(int i = 0; i < large_dataset.size(); i += BATCH_SIZE){
            size_t current_batch_size = std::min(BATCH_SIZE, large_dataset.size() - i);

            std::vector<MoleculeAtom> batch_molecules(
                large_dataset.begin() + i, 
                large_dataset.begin() + i + current_batch_size
            );

            batches.push_back(batch_molecules);
            batch_results.resize(batches.size());
        }

        std::cout << "Created " << batches.size() << " batches" << std::endl;

        for(int batch_idx = 0; batch_idx < batches.size(); batch_idx++){
            int stream_id = batch_idx % NUM_STREAMS;

            batch_results[batch_idx] = gpu::compute_affinity_AoS_async(batches[batch_idx], stream_id);
        }

        std::cout << "Synchronizing all streams..." << std::endl;
        gpu::synchronize_all_streams();
    
        for (const auto& batch_result : batch_results) {
            final_results.insert(final_results.end(), batch_result.begin(), batch_result.end());
        }
    
        gpu::cleanup_streams();

       } else if(type == 1){
        std::cout << "\n --- channel_affinity AoS ---\n";

        gpu::init_streams(NUM_STREAMS);

        std::vector<std::vector<MoleculeAtom>> batches;
        std::vector<std::vector<float>> batch_results;

        for(int i = 0; i < large_dataset.size(); i += BATCH_SIZE){
            size_t current_batch_size = std::min(BATCH_SIZE, large_dataset.size() - i);

            std::vector<MoleculeAtom> batch_molecules(
                large_dataset.begin() + i, 
                large_dataset.begin() + i + current_batch_size
            );

            batches.push_back(batch_molecules);
            batch_results.resize(batches.size());
        }

        std::cout << "Created " << batches.size() << " batches" << std::endl;

        for(int batch_idx = 0; batch_idx < batches.size(); batch_idx++){
            int stream_id = batch_idx % NUM_STREAMS;

            batch_results[batch_idx] = gpu::compute_affinity_channel_AoS_async(batches[batch_idx], stream_id);
        }

        std::cout << "Synchronizing all streams..." << std::endl;
        gpu::synchronize_all_streams();
    
        for (const auto& batch_result : batch_results) {
            final_results.insert(final_results.end(), batch_result.begin(), batch_result.end());
        }
    
        gpu::cleanup_streams();

       } else if(type == 2){
        std::cout << "\n --- TEXTURE general_affinity AoS ---\n";

        gpu::init_streams(NUM_STREAMS);

        std::vector<std::vector<MoleculeAtom>> batches;
        std::vector<std::vector<float>> batch_results;

        for(int i = 0; i < large_dataset.size(); i += BATCH_SIZE){
            size_t current_batch_size = std::min(BATCH_SIZE, large_dataset.size() - i);

            std::vector<MoleculeAtom> batch_molecules(
                large_dataset.begin() + i, 
                large_dataset.begin() + i + current_batch_size
            );

            batches.push_back(batch_molecules);
            batch_results.resize(batches.size());
        }

        std::cout << "Created " << batches.size() << " batches" << std::endl;

        for(int batch_idx = 0; batch_idx < batches.size(); batch_idx++){
            int stream_id = batch_idx % NUM_STREAMS;

            batch_results[batch_idx] = gpu::compute_affinity_texture_async(batches[batch_idx], stream_id);
        }

        std::cout << "Synchronizing all streams..." << std::endl;
        gpu::synchronize_all_streams();
    
        for (const auto& batch_result : batch_results) {
            final_results.insert(final_results.end(), batch_result.begin(), batch_result.end());
        }
    
        gpu::cleanup_streams();

       } else if(type == 3){
        // TODO
       }

    return final_results;
}

void benchmark_batch_sizes(const std::vector<MoleculeAtom>& large_dataset) {
    std::vector<size_t> batch_sizes = {10000, 25000, 50000, 100000};
    std::vector<int> stream_counts = {1, 2, 4, 8};
    
    std::cout << "\n===== BATCH SIZE BENCHMARK =====\n";
    std::cout << "Batch Size\tStreams\tTotal Time\tThroughput\n";
    
    for (auto batch_size : batch_sizes) {
        for (auto num_streams : stream_counts) {
            if (batch_size * num_streams > large_dataset.size()) continue;
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Simula processing con questi parametri
            gpu::init_streams(num_streams);
            
            size_t processed = 0;
            for (size_t i = 0; i < large_dataset.size(); i += batch_size) {
                size_t current_batch = std::min(batch_size, large_dataset.size() - i);
                
                std::vector<MoleculeAtom> batch(
                    large_dataset.begin() + i,
                    large_dataset.begin() + i + current_batch
                );
                
                int stream_id = (i / batch_size) % num_streams;
                auto result = gpu::compute_affinity_AoS_async(batch, stream_id);
                processed += current_batch;
            }
            
            gpu::synchronize_all_streams();
            gpu::cleanup_streams();
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            double throughput = (double)processed / duration.count();  // atomi/μs
            
            std::cout << batch_size << "\t\t" << num_streams << "\t" 
                      << duration.count() << "μs\t" << throughput << "\n";
        }
    }
}

int main() {
    std::string filepath_protein = "../data/pocket.geneo.csv";
    std::string filepath_molecule = "../data/6ugn_ligand.mol2.csv";
    auto molecule_atoms = parse_ligand_file(filepath_molecule);
    auto protein_atoms = parse_protein_file(filepath_protein);
    std::vector<float> result[2];

    init_grid(protein_atoms);

    result[0] = compute_affinity(molecule_atoms);
    result[1] = compute_affinity_channel(molecule_atoms);

    valutate_performance_cpu(1,molecule_atoms);
    valutate_performance_cpu(2,molecule_atoms);
 
    std::cout << "\n===== GPU PROCESSING =====\n";
    
    // Initialized Cuda enviroment
    gpu::init();
    gpu::init_grid(grid_unique, grid_size * n_channel);
    gpu::init_grid_texture(grid_unique, grid_size * n_channel);
    
/* // --- Array of Struct approach (for molecules) ---

    std::cout << "\n--- AoS Approach ---\n";
    std::vector<float> result_gpu[4];

    result_gpu[0] = gpu::compute_affinity(molecule_atoms);
    result_gpu[1] = gpu::compute_affinity_channel(molecule_atoms);
    
    gpu::evaluate_performance(1, molecule_atoms); // all channel
    gpu::evaluate_performance(2, molecule_atoms); 
    
// --- Struct of Array approach (for molecules) ---

    std::cout << "\n--- SoA Approach ---\n";
    MoleculeData molecule_data = convert_molecule_to_SoA(molecule_atoms);
    
    result_gpu[2] = gpu::compute_affinity_soa(molecule_data);
    result_gpu[3] = gpu::compute_affinity_channel_soa(molecule_data);
    
    gpu::evaluate_performance_soa(1, molecule_data);// all channel
    gpu::evaluate_performance_soa(2, molecule_data);
 */
// --- Large dataset of Molecule --- 

    std::vector<MoleculeAtom> large_dataset;

    // duplicating the same molecule
    for(int i = 0; i < dataset_size; i++){
        for(auto atom : molecule_atoms){
            MoleculeAtom new_atom = atom;
            large_dataset.push_back(new_atom);
        }
    }

    std::cout << "Generated " << large_dataset.size() << " atoms in large dataset" << std::endl;
    
    for(int i = 0; i < 4; i++){
        auto start_time = std::chrono::high_resolution_clock::now();
        auto large_results = dataset_gpu_processing(large_dataset, i); // Gpu function call
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Large dataset processing completed in " << total_time.count() << " ms" << std::endl;
        std::cout << "Results: " << large_results.size() << " values" << std::endl;
    }

 

  // benchmark_batch_sizes(large_dataset);

    
    gpu::cleanup();
    
    return 0;
}





