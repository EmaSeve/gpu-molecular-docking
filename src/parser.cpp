#include "parser.h"
#include "random.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<MoleculeAtom> parse_ligand_file(const std::string& filename) {
    std::vector<MoleculeAtom> atoms;
    std::ifstream file(filename);
    std::string line;

    do
    {
        std::getline(file,line);
    }
    while(line.compare("@<TRIPOS>ATOM"));
        
    std::getline(file,line);
    while(line.compare("@<TRIPOS>BOND")){
        std::stringstream ss(line);
        std::string field;
        MoleculeAtom atom;

        int id;
        float x,y,z,charge;
        std::string name,type,sub_id,sub_name;
        ss >> id >> name >> x >> y >> z >> type >> sub_id >> sub_name >> charge;
        
        atom.id = id;
        atom.x = x;
        atom.y = y;
        atom.z = z;
        atom.charge = charge;
        atom.channel = random_channel();

        atoms.push_back(atom);

        std::getline(file,line);
    }
    
    return atoms;
}

std::vector<ProteinAtom> parse_protein_file(const std::string& filename) {
    std::vector<ProteinAtom> atoms;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cout << "Errore: impossibile aprire il file " << filename << std::endl;
        return atoms;
    }
    // Legge e ignora l'intestazione
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line); // stringa della riga corrente
        std::string field;
        ProteinAtom atom;
        unsigned int fieldIndex = 0;
        
        // per ogni elemento della riga separato da ","
        while (std::getline(ss, field, ',')) {
            float value = std::stof(field); // converte stringa in float
            if(fieldIndex == 0) atom.id = value;
            else if (fieldIndex == 4) atom.x = value;
            else if (fieldIndex == 5) atom.y = value;
            else if (fieldIndex == 6) atom.z = value;
            else if (fieldIndex >= 8 && fieldIndex <= 15)
                atom.psi[fieldIndex - 8] = value;

            ++fieldIndex;
        }

        if (fieldIndex == 16){
            atoms.push_back(atom);
        }    
    }
    return atoms;
}
