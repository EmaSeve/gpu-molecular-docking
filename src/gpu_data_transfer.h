#ifndef GPU_DATA_TRANSFER_H
#define GPU_DATA_TRANSFER_H

#include "data_structures.h"

namespace gpu {
    // Funzioni di allocazione e trasferimento memoria per SoA
    MoleculeDataGPU convert_molecule_to_SoA_gpu(const MoleculeData& host);
    void free_molecule_gpu(MoleculeDataGPU& device);
}

#endif // GPU_DATA_TRANSFER_H