# Protein-Ligand CUDA

Efficient implementation of a protein-ligand interaction engine using CUDA acceleration and multi-layered 3D mapping.

## Features
- Parses molecular and protein input from CSV/mol2
- Discretizes atoms into a 3D grid with 8 data channels
- Computes binding affinity using CUDA-ready structures
- Supports minimum distance analysis between ligand and protein atoms

## File structure
- `src/` – C++ and CUDA source code
- `data/` – Input files
- `notes/` – Project notes and specs

## Build
```bash
g++ -std=c++17 -o main src/*.cpp
