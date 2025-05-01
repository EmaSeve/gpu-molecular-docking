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
std::unordered_map<int, ProteinAtom*> grid_map;

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

    // sposto di 0.25 perchè per ora ho il centro del voxel
    x_max += cell_size * 0.5;
    x_min -= cell_size * 0.5;

    y_max += cell_size * 0.5;
    y_min -= cell_size * 0.5;

    z_max += cell_size * 0.5;
    z_min -= cell_size * 0.5;


    X = static_cast<int>((x_max - x_min) / cell_size);
    Y = static_cast<int>((y_max - y_min) / cell_size);
    Z = static_cast<int>((z_max - z_min) / cell_size);
}

// this function compute the key by using ah hash map
// the input is the cell's id = x_cell + y_cell*10 + z_cell*100
int compute_global_index(const float& atom_x, const float& atom_y, const float& atom_z){
    
    // compute the relative cell in the grid
    int cell_x = static_cast<int>((atom_x - x_min) / cell_size);
    int cell_y = static_cast<int>((atom_y - y_min) / cell_size);
    int cell_z = static_cast<int>((atom_z - z_min) / cell_size);

    // global index of the cell
    int index = cell_x + cell_y * (X) + cell_z * (X * Y);

   
    return index;
}

std::vector<float> compute_affinity(const std::vector<MoleculeAtom> & molecule_atoms){
    int atom_index = 0;
    std::vector<float> result(molecule_atoms.size() * 8);
    for(const auto& atom : molecule_atoms){
        int grid_index = compute_global_index(atom.x,atom.y,atom.z);
        if(grid_map.find(grid_index) != grid_map.end()){
            ProteinAtom* protein_atom = grid_map[grid_index];
            for(int i = 0; i < n_channel; i++){
                result[atom_index * 8 + i] = atom.charge * protein_atom->psi[i];
            } 
        } else{
            for(int i = 0; i < n_channel; i++){
                result[atom_index * 8 + i] = 0.0;
            }
            std::cout<<"Not matched, id molecula: "<<atom.id<<std::endl;
        }
        atom_index++;   
    }
    
    return result;
}

std::vector<float> compute_min_distance(const std::vector<ProteinAtom>& protein_atoms, const std::vector<MoleculeAtom> & molecule_atoms){
    float dist;
    std::vector<float> result;

    for(const auto& ligand : molecule_atoms){
        float min_dist = std::numeric_limits<float>::infinity();
        float pre_min_dist = std::numeric_limits<float>::infinity();
        float id;

        for(const auto& protein : protein_atoms){
            dist = std::sqrt( std::pow((ligand.x - protein.x), 2) +  std::pow((ligand.y - protein.y), 2) + std::pow((ligand.z - protein.z), 2)  );
            if(dist < min_dist){
                min_dist = dist;
                id = protein.id;
            }
                 
        }

        result.push_back(min_dist);
        result.push_back(id);

    }

    return result;

}

std::vector<float> min_distance_protein(const std::vector<ProteinAtom>& protein_atoms){
    float dist;
    std::vector<float> result;

    for(const auto& ligand : protein_atoms){
        float min_dist = std::numeric_limits<float>::infinity();
        float pre_min_dist = std::numeric_limits<float>::infinity();

        for(const auto& protein : protein_atoms){
            if(protein.id != ligand.id){
                dist = std::sqrt( std::pow((ligand.x - protein.x), 2) +  std::pow((ligand.y - protein.y), 2) + std::pow((ligand.z - protein.z), 2)  );
            if(dist < pre_min_dist){
                if(dist < min_dist){
                    min_dist = dist;
                } else{
                    pre_min_dist = dist;
                }

            }
            }     
                 
        }

        result.push_back(min_dist);
        result.push_back(pre_min_dist);

    }

    return result;

}



int main() {
    std::string filepath_protein = "../data/pocket.geneo.csv";
    std::string filepath_molecule = "../data/6ugn_ligand.mol2.csv";
    auto molecule_atoms = parse_ligand_file(filepath_molecule);
    auto protein_atoms = parse_protein_file(filepath_protein);
    std::vector<float> result;

    /* for(const ProteinAtom& atom : protein_atoms){
        std::cout << "Posizione: (" << atom.x << ", " << atom.y << ", " << atom.z << ")"
        << " | Psi[0]: " << atom.psi[0] << std::endl;
    }
   
    for(const MoleculeAtom& atom : molecule_atoms){
        std::cout << "Posizione: (" << atom.x << ", " << atom.y << ", " << atom.z << ")"
                  << " | Charge: " << atom.charge << std::endl;
    } */

    compute_grid_dimension(protein_atoms);
    std::cout<<"X:"<<X<<"Y:"<<Y<<"Z:"<<Z<<std::endl;


     for(auto& atom : protein_atoms){
        int index = compute_global_index(atom.x, atom.y, atom.z);
        grid_map[index] =  &atom;
    }

    result = compute_affinity(molecule_atoms);

    for(int i = 0; i<molecule_atoms.size();i++){
        int id = i + 1;
        int grid_index = compute_global_index(molecule_atoms[i].x,molecule_atoms[i].y,molecule_atoms[i].z);
        if(grid_map.find(grid_index) != grid_map.end()){
            ProteinAtom* protein_atom_associated = grid_map[grid_index];
            std::cout<<"Atom id: "<<id<<", grid cell: "<<grid_index<<", Protein_atom id: "<<protein_atom_associated->id<<std::endl;
            for(int j = 0; j<8; j++){
                std::cout<<"  output channel "<<j+1<<": "<<result[i*8 + j]<<std::endl;
            }
        }
    }

     std::vector<float> min_distance;
    min_distance = compute_min_distance(protein_atoms,molecule_atoms);

    for(int i=0; i<min_distance.size();i=i+2){
        std::cout<<"Id: "<<i/2+1<<", min distance: "<<min_distance[i]<<" ID Protein: "<<min_distance[i+1]<<std::endl;
    }

    


    return 0;
}











