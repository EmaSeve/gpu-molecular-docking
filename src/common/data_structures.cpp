// gpu_data_transfer.cpp
#include "data_structures.h"

// Funzione di conversione per AoS -> SoA CPU
MoleculeData convert_molecule_to_SoA(const std::vector<MoleculeAtom>& molecule_atoms){
    MoleculeData molecule_data;
    int molecule_size = molecule_atoms.size();
    
    molecule_data.id.resize(molecule_size);
    molecule_data.x.resize(molecule_size);
    molecule_data.y.resize(molecule_size);
    molecule_data.z.resize(molecule_size);
    molecule_data.charge.resize(molecule_size);
    for (int j = 0; j < 8; j++) {
        molecule_data.channel[j].resize(molecule_size);
    }

    for(int i = 0; i < molecule_size; i++){
        molecule_data.id[i] = molecule_atoms[i].id;
        molecule_data.x[i] = molecule_atoms[i].x;
        molecule_data.y[i] = molecule_atoms[i].y;
        molecule_data.z[i] = molecule_atoms[i].z;
        molecule_data.charge[i] = molecule_atoms[i].charge;
        for(int j = 0; j < 8; j++){
            molecule_data.channel[j][i] = molecule_atoms[i].channel[j];
        }
    }
    return molecule_data;
}

// Funzione di conversione per AoS -> gpu_AoS
std::vector<CudaMoleculeAtom> convert_molecule_to_AoS_gpu(const std::vector<MoleculeAtom>& molecule_atoms) {
    std::vector<CudaMoleculeAtom> cuda_atoms(molecule_atoms.size());
    
    for (std::size_t i = 0; i < molecule_atoms.size(); i++) {
        cuda_atoms[i].id = molecule_atoms[i].id;
        cuda_atoms[i].x = molecule_atoms[i].x;
        cuda_atoms[i].y = molecule_atoms[i].y;
        cuda_atoms[i].z = molecule_atoms[i].z;
        cuda_atoms[i].charge = molecule_atoms[i].charge;
        
        for (int c = 0; c < 8; c++) {
            cuda_atoms[i].channel[c] = (c < molecule_atoms[i].channel.size()) ? 
                                      molecule_atoms[i].channel[c] : 0;
        }
    } 
    return cuda_atoms;
}

extern int X, Y, Z;
extern float x_min, y_min, z_min;

GridConstants convert_to_grid_constants(const int n_channel, const float cell_size) {
    GridConstants constants;

    constants.n_channel = n_channel;
    constants.cell_size = cell_size;

    constants.X = X;
    constants.Y = Y;
    constants.Z = Z;
    constants.x_min = x_min;
    constants.y_min = y_min;
    constants.z_min = z_min;
    
    return constants;
}
