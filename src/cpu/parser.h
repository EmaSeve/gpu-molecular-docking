#ifndef PARSER_H
#define PARSER_H

#include <vector>
#include <string>
#include "../common/data_structures.h"

std::vector<MoleculeAtom> parse_ligand_file(const std::string& filename);
std::vector<ProteinAtom> parse_protein_file(const std::string& filename);

#endif






