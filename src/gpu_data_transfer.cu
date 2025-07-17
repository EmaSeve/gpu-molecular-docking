#include <cuda_runtime.h>  
#include "gpu_data_transfer.h"  
#include "data_structures.h"
#include "constants.h"


namespace gpu{

// Allocate SoA in GPU memory
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

    int* temp_channel[n_channel];

    for(int i = 0; i < n_channel; i++){
        cudaMalloc(&temp_channel[i], num_atoms * sizeof(int));
        cudaMemcpy(temp_channel[i], host.channel[i].data(), num_atoms * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&device.channel, n_channel * sizeof(int*));
    cudaMemcpy(device.channel, temp_channel, n_channel * sizeof(int*), cudaMemcpyHostToDevice);

    return device;
}

// Free SoA in GPU memory
void free_molecule_gpu(MoleculeDataGPU& device) {
    cudaFree(device.id);
    cudaFree(device.x);
    cudaFree(device.y);
    cudaFree(device.z);
    cudaFree(device.charge);
    
    int* channel_host[n_channel];  
    cudaMemcpy(channel_host, device.channel, n_channel * sizeof(int*), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_channel; ++i) {
        cudaFree(channel_host[i]);
    }
    
    cudaFree(device.channel);

    device.id = nullptr;
    device.x = nullptr;
    device.y = nullptr;
    device.z = nullptr;
    device.charge = nullptr;
    device.channel = nullptr;
    
}
// used to handle async 
MoleculeDataGPU convert_molecule_to_SoA_gpu_async(const MoleculeData& host, cudaStream_t stream) {
    MoleculeDataGPU device;
    size_t num_atoms = host.id.size();

    cudaMalloc(&device.id, num_atoms * sizeof(int));
    cudaMalloc(&device.x, num_atoms * sizeof(float));
    cudaMalloc(&device.y, num_atoms * sizeof(float));
    cudaMalloc(&device.z, num_atoms * sizeof(float));
    cudaMalloc(&device.charge, num_atoms * sizeof(float));
    
    cudaMemcpyAsync(device.id, host.id.data(), num_atoms * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device.x, host.x.data(), num_atoms * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device.y, host.y.data(), num_atoms * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device.z, host.z.data(), num_atoms * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device.charge, host.charge.data(), num_atoms * sizeof(float), cudaMemcpyHostToDevice, stream);

    int* temp_channel[n_channel];

    for(int i = 0; i < n_channel; i++){
        cudaMalloc(&temp_channel[i], num_atoms * sizeof(int));
        cudaMemcpyAsync(temp_channel[i], host.channel[i].data(), num_atoms * sizeof(int), cudaMemcpyHostToDevice, stream);
    }

    cudaMalloc(&device.channel, n_channel * sizeof(int*));
    cudaMemcpyAsync(device.channel, temp_channel, n_channel * sizeof(int*), cudaMemcpyHostToDevice, stream);

    return device;
}

// Free SoA in GPU memory
void free_molecule_gpu_async(MoleculeDataGPU& device, cudaStream_t stream) {
    cudaFree(device.id);
    cudaFree(device.x);
    cudaFree(device.y);
    cudaFree(device.z);
    cudaFree(device.charge);
    
    int* channel_host[n_channel];  
    cudaMemcpyAsync(channel_host, device.channel, n_channel * sizeof(int*), cudaMemcpyDeviceToHost, stream);

    for (int i = 0; i < n_channel; ++i) {
        cudaFree(channel_host[i]);
    }
    
    cudaFree(device.channel);

    device.id = nullptr;
    device.x = nullptr;
    device.y = nullptr;
    device.z = nullptr;
    device.charge = nullptr;
    device.channel = nullptr;
    
}




}