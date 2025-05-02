import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Bio.PDB.Polypeptide import PPBuilder
import re
from Bio.PDB import PDBParser, PDBList
import numpy as np

def get_chain_sequence(pdb_id, chain_id):

    pdb_file = "PDBs/" + pdb_id + ".pdb"
    #print(pdb_file)
    #pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format="pdb")
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
# ----------------------------
# 1. Load and preprocess data
# ----------------------------
print("Here we go again....")
df = pd.read_csv("skempi_v2.csv",sep=';')  
sequences = []

ddg_values = []
for idx, row in df.iterrows():
    if ',' in row['Mutation(s)_cleaned']:
        continue
    pdb_id = row['#Pdb'][0:4]
    chain_id = row['Mutation(s)_cleaned'][1]
    wt_sequence = get_chain_sequence(pdb_id, chain_id)
    pos = int(re.search(r'[A-Za-z](\d+)[A-Za-z]', row['Mutation(s)_cleaned']).group(1))
    mt_sequence = wt_sequence[:pos-1] + row['Mutation(s)_cleaned'][0] + wt_sequence[pos:]
    full_seq = wt_sequence + " [SEP] " + mt_sequence
    if(len(full_seq)<=1024):
        continue
    sequences.append(full_seq)
    
    


    ddg = ((8.314/4184)*(273.15 + 25.0)* np.log(row['Affinity_mut_parsed'])) - ((8.314/4184)*(273.15 + 25.0)* np.log(row['Affinity_wt_parsed']))
    if pd.isna(ddg):
        ddg = 0
    
    ddg_values.append(ddg)
# Split
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_LENGTH = 1024

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,  # For regression
    problem_type="regression"
)

# Dataset class
class DDGDataset(Dataset):
    def __init__(self, sequences, ddgs):
        self.encodings = tokenizer(sequences, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        self.labels = torch.tensor(ddgs).float().unsqueeze(1)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Prepare dataset
dataset = DDGDataset(sequences, ddg_values)

# Split into train/val
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

# Training arguments
training_args = TrainingArguments(
    output_dir="./esm2_ddg_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

# Save model & tokenizer
model.save_pretrained("esm2_ddg_finetuned")
tokenizer.save_pretrained("esm2_ddg_finetuned")