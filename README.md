# Protein-Ligand CUDA

## Project Overview

This project aims to design and implement an efficient system for analyzing the interaction between a target protein and a ligand molecule, using CUDA for high-performance, parallel processing on GPUs.

The program is initially developed in C++, and later optimized with CUDA. The core idea is to represent the protein as a **3D multi-layered spatial map**, where each spatial cell (voxel) holds **8 data channels** representing physicochemical properties.

---

## Input Data

- A CSV file containing:
  - Atom coordinates `(x, y, z)` for each atom in the **target protein**
  - 8 associated values per atom (one per channel)

- A `.mol2` file (converted to CSV) containing:
  - Atom coordinates `(x, y, z)` for each atom in the **ligand molecule**
  - A **charge value** for each atom

---

## Functional Goals

1. **Spatial Discretization**  
   Atom coordinates are mapped into a **3D grid** (voxel space) based on a fixed cell size. Each voxel becomes a container for protein information.

2. **Multi-Layered Data Structure**  
   Each voxel in the grid contains **8 channels** representing different physicochemical attributes (e.g., electrostatic potential, hydrophobicity, hydrogen bond donors/acceptors).

3. **Interaction Computation**  
   For each atom in the ligand molecule:
   - Identify the corresponding cell in the protein grid
   - Compute the product between the ligand’s atomic charge and the protein’s 8 channel values
   - Store results either:
     - In 8 separate arrays (Type 1: one per channel)
     - In a single array (Type 2: use only a specific channel)

---

## Parallelization Goals (CUDA Phase)

- Optimize data structures for fast, parallel access
- Implement CUDA kernels for affinity computation and distance metrics
- Ensure memory access patterns are GPU-friendly

---

## File structure
- `src/` – C++ and CUDA source code
- `data/` – Input files
- `notes/` – Project notes and specs

## Build
```bash
g++ -std=c++17 -o main src/*.cpp
