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
    // Costanti device
    __constant__ int d_n_channel;
    __constant__ float d_cell_size;
    __constant__ int d_X, d_Y, d_Z;
    __constant__ float d_x_min, d_y_min, d_z_min;
    
    // Memoria della griglia su device
    float* d_grid_unique = nullptr;
    
    // Flag per tracciare l'inizializzazione
    bool initialized = false;
    

    __device__ int compute_cell_index(const float& atom_x, const float& atom_y, const float& atom_z){
        // compute the relative cell in the grid
        int cell_x = static_cast<int>((atom_x - d_x_min) / d_cell_size);
        int cell_y = static_cast<int>((atom_y - d_y_min) / d_cell_size);
        int cell_z = static_cast<int>((atom_z - d_z_min) / d_cell_size);

        // index of the cell
        int index = cell_x + cell_y * (d_X) + cell_z * (d_X * d_Y);

        return index;
    }


    // Kernel declarations (implementazioni omesse come richiesto)
    __global__ void compute_affinity_kernel(const CudaMoleculeAtom* molecule_atoms, 
        int num_atoms, const float* grid_unique, float* result){
                                            
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if(idx >= num_atoms) return;
            
            float x = molecule_atoms[idx].x;
            float y = molecule_atoms[idx].y;
            float z = molecule_atoms[idx].z;
            float charge = molecule_atoms[idx].charge;

            int cell_idx = compute_cell_index(x, y, z);

            for(int i=0; i < d_n_channel; i++){
                result[idx * d_n_channel + i] = charge * grid_unique[cell_idx * d_n_channel + i];
            }
    }
                                           
    __global__ void compute_affinity_channel_kernel(const CudaMoleculeAtom* molecule_atoms, 
                                                   int num_atoms,
                                                   const float* grid_unique, 
                                                   float* result);
                                                   
    __global__ void compute_affinity_kernel_soa(const int* id, const float* x, const float* y, const float* z,
        const float* charge, const float* grid_unique, float* result, int num_atoms){
            
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if(idx >= num_atoms) return;

            float x_thread = x[idx];
            float y_thread = y[idx];
            float z_thread = z[idx];
            float charge_thread = charge[idx];

            int cell_idx = compute_cell_index(x_thread,y_thread,z_thread);

            for(int i = 0; i < d_n_channel; i++){
                result[idx * d_n_channel + i] = charge_thread * grid_unique[cell_idx * d_n_channel + i];
            }
            // here we access the memory 8 times, and the access it's NOT coalescent, since depends
            // on the cell_idx, and every thread of a block can access to far position in the grid
            
            // how to deal with: 
            // - 8 access: should I load in shared memory, but if we have nothing to share since every thread need differents value
            //             I would need to store every cell of the grid needed
            // - how to manipulate the grid to have coalescent access?
            // 
    }
                                               
    __global__ void compute_affinity_channel_kernel_soa(const int* id,
                                                       const float* x, const float* y, const float* z,
                                                       const float* charge,
                                                       const int* channels[],
                                                       const float* grid_unique,
                                                       float* d_result,
                                                       int num_atoms);
    
// ----------------------------------------------------------------------------------------------------
// ------------------------------------ Initalization -------------------------------------------------
// ----------------------------------------------------------------------------------------------------
    void init() {
        if (initialized) return;
        
        // use the read only memory of GPU, that is faster
        cudaMemcpyToSymbol(d_n_channel, &n_channel, sizeof(int));
        cudaMemcpyToSymbol(d_cell_size, &cell_size, sizeof(float));
        cudaMemcpyToSymbol(d_X, &X, sizeof(int));
        cudaMemcpyToSymbol(d_Y, &Y, sizeof(int));
        cudaMemcpyToSymbol(d_Z, &Z, sizeof(int));
        cudaMemcpyToSymbol(d_x_min, &x_min, sizeof(float));
        cudaMemcpyToSymbol(d_y_min, &y_min, sizeof(float));
        cudaMemcpyToSymbol(d_z_min, &z_min, sizeof(float));

        initialized = true;
    }
    
    void init_grid(const std::vector<float>& cpu_grid_unique, int size) {
        // Assicurati che l'inizializzazione di base sia stata fatta
        init();
        
        // Libera memoria precedente se esiste
        if (d_grid_unique != nullptr) {
            cudaFree(d_grid_unique);
        }
        
        // Alloca e copia la griglia
        cudaMalloc(&d_grid_unique, size * sizeof(float));
        cudaMemcpy(d_grid_unique, cpu_grid_unique.data(), 
                   size * sizeof(float), cudaMemcpyHostToDevice);
    }
// ----------------------------------------------------------------------------------------------------
// ------------------------------------ Array of Struct -------------------------------------------------
// ----------------------------------------------------------------------------------------------------
 
    
    std::vector<float> compute_affinity(const std::vector<MoleculeAtom>& molecule_atoms) {
        // Assicurati che le strutture di base siano inizializzate
        if (!initialized || d_grid_unique == nullptr) {
            std::cerr << "Error: CUDA environment not initialized!" << std::endl;
            return {};
        }
        
        // Dimensioni
        int num_atoms = molecule_atoms.size();
        int result_size = num_atoms * n_channel;
        
        // Alloca memoria per i risultati
        std::vector<float> result(result_size);
        
        // Converti le molecole in formato CUDA-compatibile
        // Ora utilizziamo la funzione da data_structures.h
        std::vector<CudaMoleculeAtom> cuda_molecules = convert_molecule_to_AoS_gpu(molecule_atoms);
        
        // Alloca memoria device
        CudaMoleculeAtom* d_molecules;
        float* d_result;
        
        cudaMalloc(&d_molecules, num_atoms * sizeof(CudaMoleculeAtom));
        cudaMalloc(&d_result, result_size * sizeof(float));
        
        // Copia i dati sul device
        cudaMemcpy(d_molecules, cuda_molecules.data(), 
                   num_atoms * sizeof(CudaMoleculeAtom), cudaMemcpyHostToDevice);
        
        // Lancia il kernel
        int block_size = 256;
        int num_blocks = (num_atoms + block_size - 1) / block_size;
        
        compute_affinity_kernel<<<num_blocks, block_size>>>(
            d_molecules, num_atoms, d_grid_unique, d_result);
        
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        }
        
        // Recupera i risultati
        cudaMemcpy(result.data(), d_result, 
                   result_size * sizeof(float), cudaMemcpyDeviceToHost);
         
        // Libera la memoria
        cudaFree(d_molecules);
        cudaFree(d_result);
        
        return result;
    }

// ----------------------------------------------------------------------------------------------------
    // Implementazione compute_affinity_channel per AoS
    std::vector<float> compute_affinity_channel(const std::vector<MoleculeAtom>& molecule_atoms) {
        // Implementazione simile a compute_affinity
        // ma utilizzerà compute_affinity_channel_kernel
        
        // [Implementazione simile alla precedente con il kernel channel]
        return {}; // Placeholder
    }

// ----------------------------------------------------------------------------------------------------
// ------------------------------------ Struct of Array -------------------------------------------------
// ----------------------------------------------------------------------------------------------------
 
    std::vector<float> compute_affinity_soa(const MoleculeData& molecule_data) {
        // Assicurati che le strutture di base siano inizializzate
        if (!initialized || d_grid_unique == nullptr) {
            std::cerr << "Error: CUDA environment not initialized!" << std::endl;
            return {};
        }
        
        // Dimensioni
        int num_atoms = molecule_data.id.size();
        int result_size = num_atoms * n_channel;
        
        // Alloca memoria cpu per i risultati
        std::vector<float> result(result_size);
        
        // Converti SoA in formato device
        MoleculeDataGPU d_molecule_data = convert_molecule_to_SoA_gpu(molecule_data);
        
        float* d_result;
        // Alloca memoria per i risultati su gpu
        cudaMalloc(&d_result, result_size * sizeof(float));
        
        // Lancia il kernel
        int block_size = 256;
        int num_blocks = (num_atoms + block_size - 1) / block_size;
        
        compute_affinity_kernel_soa<<<num_blocks, block_size>>>(
            d_molecule_data.id, 
            d_molecule_data.x, 
            d_molecule_data.y, 
            d_molecule_data.z, 
            d_molecule_data.charge,
            d_grid_unique, d_result, num_atoms);
        
        // Sincronizzazione
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
           std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        }

        // Copia i risultati sul host
        cudaMemcpy(result.data(), d_result, result_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Libera la memoria
        free_molecule_gpu(d_molecule_data);
        cudaFree(d_result);
        
        return result;
    }
    
// ----------------------------------------------------------------------------------------------------
    // Implementazione compute_affinity_channel per SoA
    std::vector<float> compute_affinity_channel_soa(const MoleculeData& molecule_data) {
        // [Implementazione simile alla precedente con il kernel channel per SoA]
        return {}; // Placeholder
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