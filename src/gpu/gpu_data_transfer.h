#ifndef GPU_DATA_TRANSFER_H
#define GPU_DATA_TRANSFER_H

#include "../common/data_structures.h"

namespace gpu {
    // Funzioni di allocazione e trasferimento memoria per SoA
    MoleculeDataGPU convert_molecule_to_SoA_gpu(const MoleculeData& host);
    void free_molecule_gpu(MoleculeDataGPU& device);

    MoleculeDataGPU convert_molecule_to_SoA_gpu_async(const MoleculeData& host, cudaStream_t stream);
    void free_molecule_gpu_async(MoleculeDataGPU& device, cudaStream_t stream);
}

#endif // GPU_DATA_TRANSFER_H