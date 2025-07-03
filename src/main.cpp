#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>

#include <chrono>
#include <random>


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

int main() {
    std::string filepath_protein = "../data/pocket.geneo.csv";
    std::string filepath_molecule = "../data/6ugn_ligand.mol2.csv";
    auto molecule_atoms = parse_ligand_file(filepath_molecule);
    auto protein_atoms = parse_protein_file(filepath_protein);
    std::vector<float> result;

    init_grid(protein_atoms);

    result = compute_affinity(molecule_atoms);
    result = compute_affinity_channel(molecule_atoms);

    valutate_performance_cpu(1,molecule_atoms);
    valutate_performance_cpu(2,molecule_atoms);
 
    std::cout << "\n===== GPU PROCESSING =====\n";
    
    // Initialized Cuda enviroment
    gpu::init();
    gpu::init_grid(grid_unique, grid_size * n_channel);
    
// --- Array of Struct approach (for molecules) ---

    std::cout << "\n--- AoS Approach ---\n";
    std::vector<float> result_gpu;

    result_gpu = gpu::compute_affinity(molecule_atoms);
    result_gpu = gpu::compute_affinity_channel(molecule_atoms);
    
    gpu::evaluate_performance(1, molecule_atoms); // all channel
    gpu::evaluate_performance(2, molecule_atoms);
    
// --- Struct of Array approach (for molecules) ---

    std::cout << "\n--- SoA Approach ---\n";
    MoleculeData molecule_data = convert_molecule_to_SoA(molecule_atoms); // create the AoS data structure
    
    result_gpu = gpu::compute_affinity_soa(molecule_data);
    result_gpu = gpu::compute_affinity_channel_soa(molecule_data);
    
    gpu::evaluate_performance_soa(1, molecule_data);// all channel
    gpu::evaluate_performance_soa(2, molecule_data);
    
    gpu::cleanup();

    return 0;
}











