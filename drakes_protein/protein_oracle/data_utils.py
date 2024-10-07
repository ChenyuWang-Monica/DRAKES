import torch
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser
import os
import numpy as np

MASKED_TOKEN = 'Z'
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
ALPHABET_WITH_MASK = ALPHABET + MASKED_TOKEN
MASK_TOKEN_INDEX = ALPHABET_WITH_MASK.index(MASKED_TOKEN)


class ProteinStructureDataset(Dataset):
    def __init__(self, directory, max_len):
        """
        Args:
            directory (string): Directory with all the .pdb files.
            max_len (int): Maximum length of the protein sequences.
        """
        self.directory = directory
        self.max_len = max_len
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.pdb')]
        self.parser = PDBParser(QUIET=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        structure = self.parser.get_structure(id=None, file=file_path)
        model = structure[0]  # Assuming only one model per PDB file

        # Extract coordinates for N, CA, C, O atoms for each residue
        coords = []
        for residue in model.get_residues():
            try:
                n = residue['N'].get_coord()
                ca = residue['CA'].get_coord()
                c = residue['C'].get_coord()
                o = residue['O'].get_coord()
                coords.append([n, ca, c, o])
            except KeyError:
                continue  # Skip residues that do not have all atoms

        # Pad the coordinates to max_len
        coords = np.array(coords, dtype=np.float32)
        if len(coords) < self.max_len:
            padding = np.zeros((self.max_len - len(coords), 4, 3), dtype=np.float32)
            coords = np.concatenate((coords, padding), axis=0)
        elif len(coords) > self.max_len:
            print(f"Protein sequence in {self.filenames[idx]} is longer than max_len. Truncating.")
            coords = coords[:self.max_len]

        return torch.tensor(coords), self.filenames[idx]


class ProteinDPODataset(Dataset):
    def __init__(self, dpo_train_dict, pdb_idx_dict, pdb_structure):
        self.dpo_train_dict = dpo_train_dict
        self.protein_list = list(dpo_train_dict.keys())
        self.pdb_idx_dict = pdb_idx_dict
        self.pdb_structure = pdb_structure

    def __len__(self):
        return len(self.protein_list)
    
    def __getitem__(self, idx):
        protein_name = self.protein_list[idx]
        protein_data = self.dpo_train_dict[protein_name]
        protein_structure = self.pdb_structure[self.pdb_idx_dict[protein_data[1]]]
        
        return {
            'protein_name': protein_name,
            'aa_seq': protein_data[0],
            'WT_name': protein_data[1],
            'aa_seq_wt': protein_data[2],
            'dG_ML': protein_data[3],
            'dG_ML_wt': protein_data[5],
            'name_wt': protein_data[7],
            'structure': protein_structure,
            }


def featurize(batch, device):
    B = batch['structure'].shape[0]
    L_max = max([len(x) for x in batch['aa_seq']])
    X = batch['structure'][:, :L_max, :, :].to(dtype=torch.float32, device=device)
    S = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    S_wt = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.int32)
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    for i, seq in enumerate(batch['aa_seq']):
        S[i, :len(seq)] = np.asarray([ALPHABET.index(aa) for aa in seq], dtype=np.int32)
        mask[i, :len(seq)] = 1
        residue_idx[i, :len(seq)] = np.arange(len(seq))
    for i, seq in enumerate(batch['aa_seq_wt']):
        S_wt[i, :len(seq)] = np.asarray([ALPHABET.index(aa) for aa in seq], dtype=np.int32)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    S_wt = torch.from_numpy(S_wt).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    chain_M = mask.clone()
    chain_encoding_all = mask.clone()
    return X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt

