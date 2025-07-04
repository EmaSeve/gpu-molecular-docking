#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "data_structures.h"
#include "gpu_functions.h"
#include "gpu_data_transfer.h"
#include "constants.h"

// Riferimenti alle variabili globali da main.cpp
extern int X, Y, Z;
extern float x_min, y_min, z_min;
extern int grid_size;

// Variabili globali device
namespace gpu {

    __constant__ GridConstants d_constants;
    
    float* d_grid_unique = nullptr;
    
    // flag to trace initialization
    bool initialized = false;
    
    // compute the index of the cell in grid_unique, based on the coordinates of the atom
    __device__ int compute_cell_index(const float& atom_x, const float& atom_y, const float& atom_z){
        // relative cell in the grid
        int cell_x = static_cast<int>((atom_x - d_constants.x_min) / d_constants.cell_size);
        int cell_y = static_cast<int>((atom_y - d_constants.y_min) / d_constants.cell_size);
        int cell_z = static_cast<int>((atom_z - d_constants.z_min) / d_constants.cell_size);

        // index of the cell
        int index = cell_x + cell_y * (d_constants.X) + cell_z * (d_constants.X * d_constants.Y);

        return index;
    }


// ----------------------------------------------------------------------------------------------------
// ------------------------------------ Kernel Functions -------------------------------------------------
// ----------------------------------------------------------------------------------------------------
  
    __global__ void compute_affinity_kernel(const CudaMoleculeAtom* molecule_atoms, 
        int num_atoms, const float* grid_unique, float* result){
             
            extern __shared__ float shared_grid[];

            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if(idx >= num_atoms) return;
            
            float x = molecule_atoms[idx].x;
            float y = molecule_atoms[idx].y;
            float z = molecule_atoms[idx].z;
            float charge = molecule_atoms[idx].charge;

            int cell_idx = compute_cell_index(x, y, z);

            for(int i = 0; i < d_constants.n_channel; i++){
                shared_grid[threadIdx.x * d_constants.n_channel + i] = grid_unique[cell_idx * d_constants.n_channel + i];
            }

            for(int i=0; i < d_constants.n_channel; i++){
                result[idx * d_constants.n_channel + i] = charge * shared_grid[threadIdx.x * d_constants.n_channel + i];
            }

    }
                                           
    __global__ void compute_affinity_channel_kernel(const CudaMoleculeAtom* molecule_atoms, int num_atoms, const float* grid_unique,
        float* result){
            
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if(idx >= num_atoms) return;

            float x = molecule_atoms[idx].x;
            float y = molecule_atoms[idx].y;
            float z = molecule_atoms[idx].z;
            float charge = molecule_atoms[idx].charge;

            int channel = -1;

            int cell_idx = compute_cell_index(x, y, z);

            for(int i = 0; i < d_constants.n_channel; i++){
                if(molecule_atoms[idx].channel[i] == 1)
                    channel = i;
            }

            if(channel == -1) return;
            
            result[idx] = grid_unique[cell_idx * d_constants.n_channel + channel];
        }
                                                   
    __global__ void compute_affinity_kernel_soa(const int* id, const float* x, const float* y, const float* z,
        const float* charge, const float* grid_unique, float* result, int num_atoms){
            
            extern __shared__ float shared_grid[];

            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if(idx >= num_atoms) return;

            float x_thread = x[idx];
            float y_thread = y[idx];
            float z_thread = z[idx];
            float charge_thread = charge[idx];

            int cell_idx = compute_cell_index(x_thread,y_thread,z_thread);
            
            for(int i = 0; i < d_constants.n_channel; i++){
                shared_grid[threadIdx.x * d_constants.n_channel + i] = grid_unique[cell_idx * d_constants.n_channel + i];
            }

            for(int i = 0; i < d_constants.n_channel; i++){
                result[idx * d_constants.n_channel + i] = charge_thread * shared_grid[threadIdx.x * d_constants.n_channel + i];
            }
            // here we access the memory 8 times, and the access it's NOT coalescent, since depends
            // on the cell_idx, and every thread of a block can access to far position in the grid
            
            // how to deal with: 
            // - 8 access: should I load in shared memory, but if we have nothing to share since every thread need differents value
            //             I would need to store every cell of the grid needed
            // - how to manipulate the grid to have coalescent access?
            // 
    }
                                               
    __global__ void compute_affinity_channel_kernel_soa(const int* id, const float* x, const float* y, const float* z, 
        const float* charge, const int* channel, const float* grid_unique, float* result, int num_atoms){

            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if(idx >= num_atoms) return;
            
            int k = 0;
            int* p = nullptr;
            int channel_thread = -1;
            
               /*  for(int i = 0; i<8;i++){
                    p = channel[i];
                    printf("\n accessed pointer i=%d",i);
                }

                for(int i = 0; i<num_atoms;i++){
                    for(int j = 0;j<8;j++){
                        k = channel[j][i];
                         if(i==idx && k==1){
                            channel_thread = i;
                            printf("\n idx:%d, value:%d, channel:%d, atoms:%d",idx,k,j,i);
                        } 
                        printf("value of k:%d",k);    
                            
                        printf("\n accessed channel i=%d, idx_atoms=%d",j,i);
                    }
                } */

            if(idx == 0){
                k = channel[0][0];
                printf("\nk_00:%d",k);
            }


            
           
            float x_thread = x[idx];
            float y_thread = y[idx];
            float z_thread = z[idx];
            float charge_thread = charge[idx];
            

            int cell_idx = compute_cell_index(x_thread,y_thread,z_thread);

            /* printf("\nidx:%d ",idx);
            for(int i = 0; i < 8; i++) {    
                if(channel[i][idx] == 1){
                    channel_thread = i;
                    break;
                }
            }  */

            if(channel_thread == -1) return;

            result[idx] = grid_unique[cell_idx * d_constants.n_channel + channel_thread];

        }
    
// ----------------------------------------------------------------------------------------------------
// ------------------------------------ Initalization -------------------------------------------------
// ----------------------------------------------------------------------------------------------------
    void init() {
        if (initialized) return;
        
        // used the read only memory of GPU (synchronized version)
        GridConstants h_constants = convert_to_grid_constants(n_channel, cell_size);
        cudaMemcpyToSymbol(d_constants, &h_constants, sizeof(GridConstants));

        initialized = true;
    }
    
    void init_grid(const std::vector<float>& cpu_grid_unique, int size) {
        init();
        
        // free memory if already exist
        if (d_grid_unique != nullptr) {
            cudaFree(d_grid_unique);
        }
        
        // copy of the grid
        cudaMalloc(&d_grid_unique, size * sizeof(float));
        cudaMemcpy(d_grid_unique, cpu_grid_unique.data(), 
                   size * sizeof(float), cudaMemcpyHostToDevice);
    }
// ----------------------------------------------------------------------------------------------------
// ------------------------------------ Array of Struct -------------------------------------------------
// ----------------------------------------------------------------------------------------------------
 
    
    std::vector<float> compute_affinity(const std::vector<MoleculeAtom>& molecule_atoms) {
        // check inizialization
        if (!initialized || d_grid_unique == nullptr) {
            std::cerr << "Error: CUDA environment not initialized!" << std::endl;
            return {};
        }
        
        // dimension
        int num_atoms = molecule_atoms.size();
        int result_size = num_atoms * n_channel;
        
        std::vector<float> result(result_size);
        
        // Converti le molecole in formato CUDA-compatibile
        std::vector<CudaMoleculeAtom> cuda_molecules = convert_molecule_to_AoS_gpu(molecule_atoms);
        
        // devide memory of molecules
        CudaMoleculeAtom* d_molecules;
        float* d_result;
        
        cudaMalloc(&d_molecules, num_atoms * sizeof(CudaMoleculeAtom));
        cudaMalloc(&d_result, result_size * sizeof(float));
        
        // copy from host to device
        cudaMemcpy(d_molecules, cuda_molecules.data(), 
                   num_atoms * sizeof(CudaMoleculeAtom), cudaMemcpyHostToDevice);
        
        // launch kernel
        int block_size = 256;
        int num_blocks = (num_atoms + block_size - 1) / block_size;
        int shared_mem_size = block_size * n_channel * sizeof(float);

        compute_affinity_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            d_molecules, num_atoms, d_grid_unique, d_result);
        
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        }
        
        cudaMemcpy(result.data(), d_result, 
                   result_size * sizeof(float), cudaMemcpyDeviceToHost);
         
        // free gpu memory
        cudaFree(d_molecules);
        cudaFree(d_result);
        
        return result;
    }

// ----------------------------------------------------------------------------------------------------
    // Implementazione compute_affinity_channel per AoS
    std::vector<float> compute_affinity_channel(const std::vector<MoleculeAtom>& molecule_atoms) {
        if (!initialized || d_grid_unique == nullptr) {
            std::cerr << "Error: CUDA environment not initialized!" << std::endl;
            return {};
        }

        int num_atoms = molecule_atoms.size();
        int result_size = num_atoms;

        std::vector<float> result(result_size);

        std::vector<CudaMoleculeAtom> cuda_molecules = convert_molecule_to_AoS_gpu(molecule_atoms);

        float* d_result;
        CudaMoleculeAtom* d_molecules;
        
        cudaMalloc(&d_molecules, num_atoms * sizeof(CudaMoleculeAtom));
        cudaMalloc(&d_result, result_size * sizeof(float));

        cudaMemcpy(d_molecules, cuda_molecules.data(), 
            num_atoms * sizeof(CudaMoleculeAtom), cudaMemcpyHostToDevice);

        int block_size = 256;
        int num_blocks = (num_atoms + block_size-1)/ block_size;
        // int shared_mem_size = block_size * n_channel * sizeof(float); not needed in this case

        compute_affinity_channel_kernel<<<num_blocks, block_size>>>(
            d_molecules, num_atoms, d_grid_unique, d_result);
        
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        }
        
        cudaMemcpy(result.data(), d_result, 
                   result_size * sizeof(float), cudaMemcpyDeviceToHost);
         
        // free gpu memory
        cudaFree(d_molecules);
        cudaFree(d_result);

        return result; 
    }

// ----------------------------------------------------------------------------------------------------
// ------------------------------------ Struct of Array -------------------------------------------------
// ----------------------------------------------------------------------------------------------------
 
    std::vector<float> compute_affinity_soa(const MoleculeData& molecule_data) {
        // check inizialization
        if (!initialized || d_grid_unique == nullptr) {
            std::cerr << "Error: CUDA environment not initialized!" << std::endl;
            return {};
        }
        
        int num_atoms = molecule_data.id.size();
        int result_size = num_atoms * n_channel;
        
        std::vector<float> result(result_size);
        
        // Converti SoA in formato device
        MoleculeDataGPU d_molecule_data = convert_molecule_to_SoA_gpu(molecule_data);
        
        float* d_result;
        // result memory on gpu
        cudaMalloc(&d_result, result_size * sizeof(float));
        
        // launch kernel
        int block_size = 256;
        int num_blocks = (num_atoms + block_size - 1) / block_size;
        int shared_mem_size = block_size * n_channel * sizeof(float);

        compute_affinity_kernel_soa<<<num_blocks, block_size, shared_mem_size>>>(
            d_molecule_data.id, 
            d_molecule_data.x, 
            d_molecule_data.y, 
            d_molecule_data.z, 
            d_molecule_data.charge,
            d_grid_unique, d_result, num_atoms);
        
        // Synchronization
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
           std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        }

        // copy from device to host
        cudaMemcpy(result.data(), d_result, result_size * sizeof(float), cudaMemcpyDeviceToHost);

        // free gpu memory
        free_molecule_gpu(d_molecule_data);
        cudaFree(d_result);
        
        return result;
    }
    
// ----------------------------------------------------------------------------------------------------
    // Implementazione compute_affinity_channel per SoA
    std::vector<float> compute_affinity_channel_soa(const MoleculeData& molecule_data) {
       
        if (!initialized || d_grid_unique == nullptr) {
            std::cerr << "Error: CUDA environment not initialized!" << std::endl;
            return {};
        }
        int num_atoms = molecule_data.id.size();
        int result_size = num_atoms;

        std::vector<float> result(result_size);

        for(int i = 0; i < molecule_data.id.size();i++){
            for(int j = 0;j<8;j++){
                std::cout<<"channel: "<<j<<", value:"<<molecule_data.channel[j][i]<<std::endl;
            }
        }

        MoleculeDataGPU d_molecule_data = convert_molecule_to_SoA_gpu(molecule_data);

        float* d_result;
        cudaMalloc(&d_result, result_size * sizeof(float));

        int block_size = 256;
        int num_blocks = (num_atoms + block_size - 1) / block_size;
       
        compute_affinity_channel_kernel_soa<<<num_blocks, block_size>>>(
            d_molecule_data.id, 
            d_molecule_data.x, 
            d_molecule_data.y, 
            d_molecule_data.z, 
            d_molecule_data.charge,
            d_molecule_data.channel,
            d_grid_unique, d_result, num_atoms);
        
        // Synchronization
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
           std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        }

        // copy from device to host
        cudaMemcpy(result.data(), d_result, result_size * sizeof(float), cudaMemcpyDeviceToHost);

        // free gpu memory
        free_molecule_gpu(d_molecule_data);
        cudaFree(d_result);

        return result; 
    }
    
// ----------------------------------------------------------------------------------------------------
// ------------------------------------ BenchMarking -------------------------------------------------
// ----------------------------------------------------------------------------------------------------
 
    void evaluate_performance(const int type, const std::vector<MoleculeAtom>& molecule_atoms) {
        const int num_runs = 100;
        std::chrono::duration<double> total_time(0);
        std::vector<float> result;
        
        for(int i = 0; i < num_runs; i++) {
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
                  " affinity on GPU (AoS): " << average_time << " s" << std::endl;
    }
    
    void evaluate_performance_soa(const int type, const MoleculeData& molecule_data) {
        const int num_runs = 100;
        std::chrono::duration<double> total_time(0);
        std::vector<float> result;
        
        for(int i = 0; i < num_runs; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            if(type == 1) {
                result = compute_affinity_soa(molecule_data);
            } else if(type == 2) {
                result = compute_affinity_channel_soa(molecule_data);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            total_time += end - start;
        }
        
        double average_time = total_time.count() / num_runs;
        std::cout << "Average time used to compute " << 
                  (type == 1 ? "general" : "channel") << 
                  " affinity on GPU (SoA): " << average_time << " s" << std::endl;
    }

    void cleanup() {
        if (d_grid_unique != nullptr) {
            cudaFree(d_grid_unique);
            d_grid_unique = nullptr;
        }
        initialized = false;
    }



} // namespace gpu