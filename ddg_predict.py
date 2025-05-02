#!/usr/bin/env python
import argparse
import os
import sys
import re
import torch
from torch import nn
import pickle
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser, PDBList, Select, DSSP
from Bio.PDB.Polypeptide import PPBuilder, is_aa
import warnings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
warnings.filterwarnings("ignore")



#esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
#tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

esm_model = AutoModelForSequenceClassification.from_pretrained("esm2_ddg_finetuned")
tokenizer = AutoTokenizer.from_pretrained("esm2_ddg_finetuned")
esm_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_model.to(device)



# Deep learning model
class DdGPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DdGPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Add batch normalization
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Add batch normalization
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),   # Add batch normalization
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict protein binding ΔΔG from mutation.')
    parser.add_argument('--pdb', type=str, required=True, help='Path to PDB file')
    parser.add_argument('--mutation', type=str, required=True, 
                        help='Mutation in format like "A123G" (wild-type, position, mutant)')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model pickle file')
    parser.add_argument('--chain', type=str, default=None, help='Chain ID (optional, can be inferred from mutation)')
    parser.add_argument('--output', type=str, default='prediction_result.txt', help='Output file for prediction result')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--mt_af', type=str, default=None, help='Chain ID (optional, can be inferred from mutation)')
    parser.add_argument('--wt_af', type=str, default=None, help='Chain ID (optional, can be inferred from mutation)')
    
    return parser.parse_args()

def get_chain_sequence(pdb_id, chain_id):
    pdb_file = "PDBs/" + pdb_id + ".pdb"
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)

    
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                ppb = PPBuilder()
                sequence = ""
                for pp in ppb.build_peptides(chain):
                    sequence += str(pp.get_sequence())
                return sequence
    return None

def parse_mutation(mutation_str):
    """Parse mutation string to extract wild-type, position, and mutant residues."""
    # Regular expression for mutation format like A123G
    pattern = r"([A-Z])(\d+)([A-Z])"
    match = re.match(pattern, mutation_str)
    
    if not match:
        print(f"Error: Invalid mutation format '{mutation_str}'. Expected format like 'A123G'")
        sys.exit(1)
    
    wt, pos, mt = match.groups()
    pos = int(pos)  # Convert position to integer
    
    return wt, pos, mt

def verify_mutation(sequence, wt, pos):
    """Verify that the wild-type residue matches the sequence at the given position."""
    # Check if position is within sequence bounds
    if pos <= 0 or pos > len(sequence):
        print(f"Error: Position {pos} is outside sequence bounds (1-{len(sequence)})")
        sys.exit(1)
    
    # Check if wild-type matches
    actual_residue = sequence[pos-1]  # Convert to 0-indexed
    if actual_residue != wt:
        print(f"Warning: Expected {wt} at position {pos}, but found {actual_residue}")
        return False
    
    return True


def get_combined_features(wt_sequence, mutation, position, pdb_file, chain_id,wt):
    """Combine ESM embeddings with structure-based features."""
    # Parse mutation format

    wt = wt
    mt = mutation
    pos = position  # 1-indexed position
    

    
    # Apply mutation
    mt_sequence = wt_sequence[:pos-1] + mt + wt_sequence[pos:]
    full_seq = wt_sequence + " [SEP] " + mt_sequence

    
    # Calculate embedding difference
    embedding_diff = get_esm_embedding(full_seq)

    
    # Combine all features
    structure_features = []
    
    # Add simple physicochemical features about the mutation
    # Hydrophobicity scale (Kyte-Doolittle)
    hydrophobicity = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
        'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    # Volume (Å³)
    volume = {
        'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8, 
        'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6, 
        'M': 162.9, 'F': 189.9, 'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8, 
        'Y': 193.6, 'V': 140.0
    }
    
    # Add mutation-specific features
    structure_features.append(hydrophobicity.get(wt, 0))
    structure_features.append(hydrophobicity.get(mt, 0))
    structure_features.append(hydrophobicity.get(mt, 0) - hydrophobicity.get(wt, 0))
    structure_features.append(volume.get(wt, 0))
    structure_features.append(volume.get(mt, 0))
    structure_features.append(volume.get(mt, 0) - volume.get(wt, 0))
    
    
    # Convert structure features to numpy array
    structure_features = np.array(structure_features, dtype=np.float32)
    
    # Combine with embedding difference
    combined_features = np.concatenate([embedding_diff, structure_features])
    
    return combined_features


def get_esm_embedding(seq: str) -> torch.Tensor:
    """
    Generate mean-pooled embedding from a WT + [SEP] + MT sequence.

    Args:
        seq (str): Input sequence like "WT [SEP] MT"

    Returns:
        torch.Tensor: Mean-pooled embedding vector
    """
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = esm_model.base_model(**inputs, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_dim]

        mask = inputs['attention_mask'].unsqueeze(-1)  # [1, seq_len, 1]
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1)
        mean_embedding = (summed / counts).squeeze(0)  # Shape: [hidden_dim]

        return mean_embedding




def predict_ddg(feature_vector,mdl):
    """
    Predict ΔΔG for a given mutation using the trained model.
    
    Args:
        wt_sequence (str): Wild-type protein sequence
        mutation (str): Mutation in format 'A123G'
        model_path (str): Path to saved model
    
    Returns:
        float: Predicted ΔΔG value
    """
    # Load model
    checkpoint = torch.load(mdl)
    input_dim = checkpoint['input_dim']
    
    with open('ddg_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    model = DdGPredictor(input_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    torch.set_grad_enabled(False)

    feature_vector = scaler.transform(feature_vector)

    # Make prediction
    with torch.no_grad():
        feature_tensor = torch.FloatTensor(feature_vector).to(device)
    prediction = model(feature_tensor).item()

    return prediction

def main():
    # Parse command line arguments
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)
    

    wt, pos, mt = parse_mutation(args.mutation)
    
    # Determine chain ID if not provided
    chain_id = args.chain
    
    # Extract protein sequence
    wt_sequence = get_chain_sequence(args.pdb, chain_id)
    
    if args.verbose:
        print(f"Chain {chain_id} sequence: {wt_sequence[:20]}... ({len(wt_sequence)} residues)")
    
    # Verify the wild-type residue
    if not verify_mutation(wt_sequence, wt, pos):
        print("Warning: Wild-type residue mismatch! Continuing anyway...")
    
    
    
    # Extract features
    if args.verbose:
        print("Extracting features...")
    
    with open('ddg_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    feat = []
    pdb_file = "PDBs/" + args.pdb + ".pdb"

    feature_vector = get_combined_features(wt_sequence,mt,pos,pdb_file,chain_id,wt)
    feat.append(feature_vector)
    X = np.array(feat)

    # Make prediction
    if args.verbose:
        print("Making prediction...")

           
    prediction = predict_ddg(X,args.model)

    
    # Print result
    print(f"\nPrediction Results:")
    print(f"Mutation: {args.mutation}")
    print(f"Predicted ΔΔG: {prediction:.3f} kcal/mol")
    if(args.mt_af != None and args.wt_af != None):
        mt_dg = float(args.mt_af)
        wt_dg = float(args.wt_af)
        actual = ((8.314/4184)*(273.15 + 25.0)* np.log(mt_dg)) - ((8.314/4184)*(273.15 + 25.0)* np.log(wt_dg))
        print(f"Actual ΔΔG: {actual:.3f} kcal/mol")
    
    # Write to output file
    with open(args.output, 'w') as f:
        f.write(f"Mutation: {args.mutation}\n")
        f.write(f"PDB file: {args.pdb}\n")
        f.write(f"Chain: {chain_id}\n")
        f.write(f"Position: {pos}\n")
        f.write(f"Wild-type: {wt}\n")
        f.write(f"Mutant: {mt}\n")
        f.write(f"Predicted ΔΔG: {prediction:.3f} kcal/mol\n")
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
