#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <unordered_map>
#include <cmath>

#include "data_structures.h"
#include "parser.h"

constexpr int n_channel = 8;
constexpr float cell_size = 1.25;
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

// this function compute the key by using an hash map
// the input is the cell's id = x_cell + y_cell*10 + z_cell*100
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

    std::cout<<"grid_init"<<std::endl;

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

    std::cout<<"compute result"<<std::endl;
    
    return result;
}

void print_result(const std::vector<MoleculeAtom> &molecule_atoms, const std::vector<float> &result){
    for(int i = 0; i<molecule_atoms.size();i++){
        int id = i + 1;
        int cell_index = compute_cell_index(molecule_atoms[i].x,molecule_atoms[i].y,molecule_atoms[i].z);
        
        std::cout<<"Atom id: "<<id<<", grid cell: "<<cell_index<<std::endl;
        for(int j = 0; j < n_channel; j++){
            std::cout<<"  output channel "<<j+1<<": "<<result[i * n_channel + j]<<std::endl;
        }
    }
}

int main() {
    std::string filepath_protein = "../data/pocket.geneo.csv";
    std::string filepath_molecule = "../data/6ugn_ligand.mol2.csv";
    auto molecule_atoms = parse_ligand_file(filepath_molecule);
    auto protein_atoms = parse_protein_file(filepath_protein);
    std::vector<float> result;

    // initialization of the 3D grid
    init_grid(protein_atoms);

    // compute output
    result = compute_affinity(molecule_atoms);

    print_result(molecule_atoms,result);

    return 0;
}











