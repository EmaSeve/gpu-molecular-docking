# ACA Project: GPU Molecular Docking

This project is a GPU-accelerated molecular docking application developed for the Advanced Computer Architecture course. The primary goal is to optimize the affinity calculation between a target protein and a ligand molecule using C++ and CUDA.

## Project Description

The program simulates the molecular docking process by performing the following steps:

1.  **Input Processing**: It reads two `.csv` files:
    *   `pocket.geneo.csv`: Contains the coordinates `(x, y, z)` and values for 8 channels for each atom of the target protein.
    *   `6ugn_ligand.mol2.csv`: Contains the coordinates `(x, y, z)` and type for each atom of the ligand molecule.

2.  **Grid Discretization**: The protein's atom positions are mapped onto a uniform 3D grid. Each cell in this grid stores the corresponding 8-channel values, effectively creating a spatial representation of the protein's properties.

3.  **Affinity Calculation**: For each atom in the ligand, its relative position within the protein grid is calculated. A product is then computed between the protein's grid cell values and the ligand atom's values. Two affinity calculation modes are implemented:
    *   **General Affinity**: Calculates a score for each of the 8 channels.
    *   **Channel Affinity**: Calculates a score based on a single, specific channel determined by the atom's type.

4.  **GPU Optimization**: The computationally intensive affinity calculation is parallelized and optimized for execution on NVIDIA GPUs using CUDA. Key optimizations explored in this project include:
    *   **Data Layouts**: Comparison between Array of Structs (AoS) and Struct of Arrays (SoA) for molecule data representation.
    *   **Asynchronous Execution**: Use of CUDA streams to overlap data transfers and kernel execution.
    *   **Memory Hierarchy**: Use of Texture Memory for read-only access to the protein grid, exploiting its caching capabilities.
    *   **Batch Processing**: Handling large datasets of ligands by processing them in smaller, concurrent batches.

## Project Structure

```
.
├── CMakeLists.txt          # Configuration file for building with CMake.
├── README.md               # Project documentation.
├── data/                   # Contains input data files (e.g., CSV).
└── src/                    # Project source code.
    ├── common/             # Shared constants and data structures.
    │   ├── constants.cpp
    │   ├── constants.h
    │   ├── data_structures.cpp
    │   └── data_structures.h
    │
    ├── cpu/                # CPU-related modules.
    │   ├── parser.cpp
    │   ├── parser.h
    │   ├── random.cpp
    │   └── random.h
    │
    ├── gpu/                # GPU-related modules.
    │   ├── gpu_data_transfer.cu
    │   ├── gpu_data_transfer.h
    │   ├── gpu_functions.cu
    │   └── gpu_functions.h
    │
    └── main.cpp            

```

## Source Code Overview

*   `main.cpp`: Contains the main application logic, including CPU-based calculations, performance evaluation, and orchestration of GPU tasks. It also manages the processing of large datasets using batching and streams.
*   `parser.h`/`.cpp`: Implements functions for parsing the input `.csv` files for the protein and ligand.
*   `data_structures.h`/`.cpp`: Defines the core data structures used throughout the project, including layouts for both host (CPU) and device (GPU) and for both AoS and SoA patterns.
*   `gpu_functions.h`/`.cu`: Declares and defines the CUDA kernels for affinity calculation and the wrapper functions for managing GPU resources (memory, streams, etc.).
*   `gpu_data_transfer.h`/`.cu`: Handles the data transfers between host and device memory.
*   `constants.h`/`.cpp`: Defines global constants used in the project.
*   `random.h`/`.cpp`: Utility functions for random number generation.

## Prerequisites

To build and run this project, you need the following tools:

*   **CMake** (version >= 3.20)
*   **A C++17 Compiler** (e.g., GCC, Clang)
*   **NVIDIA CUDA Toolkit** (the project has been tested with version 11.x and later)

### Profiling

To analyze the performance of the CUDA code, you can use **NVIDIA Nsight Systems**. For example, to start a profiling session:

```bash
nsys profile -o gpu_profiling ./main
```

