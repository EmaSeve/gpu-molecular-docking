#include <cuda_runtime.h>  
#include "gpu_data_transfer.h"  
#include "data_structures.h"

namespace gpu{

// allocazione SoA in gpu
MoleculeDataGPU convert_molecule_to_SoA_gpu(const MoleculeData& host) {
    MoleculeDataGPU device;
    size_t num_atoms = host.id.size();

    cudaMalloc(&device.id, num_atoms * sizeof(int));
    cudaMalloc(&device.x, num_atoms * sizeof(float));
    cudaMalloc(&device.y, num_atoms * sizeof(float));
    cudaMalloc(&device.z, num_atoms * sizeof(float));
    cudaMalloc(&device.charge, num_atoms * sizeof(float));
    
    cudaMemcpy(device.id, host.id.data(), num_atoms * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device.x, host.x.data(), num_atoms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device.y, host.y.data(), num_atoms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device.z, host.z.data(), num_atoms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device.charge, host.charge.data(), num_atoms * sizeof(float), cudaMemcpyHostToDevice);

    for(int i = 0; i < 8; i++){
        cudaMalloc(&device.channel[i], num_atoms * sizeof(int));
        cudaMemcpy(device.channel[i], host.channel[i].data(), num_atoms * sizeof(int), cudaMemcpyHostToDevice);
    }

    return device;
}

// free SoA memory gpu
void free_molecule_gpu(MoleculeDataGPU& device) {
    cudaFree(device.id);
    cudaFree(device.x);
    cudaFree(device.y);
    cudaFree(device.z);
    cudaFree(device.charge);
    
    for (int i = 0; i < 8; i++) {
        cudaFree(device.channel[i]);
    }
    
    // Opzionale: imposta i puntatori a nullptr per evitare double-free
    device.id = nullptr;
    device.x = nullptr;
    device.y = nullptr;
    device.z = nullptr;
    device.charge = nullptr;
    
    for (int i = 0; i < 8; i++) {
        device.channel[i] = nullptr;
    }
}
}