import wandb
import grelu
import pandas as pd
from grelu.lightning import LightningModel
import grelu.data.dataset
import os

base_path = '/data/scratch/wangchy/seqft/'
df = pd.read_csv(os.path.join(base_path, 'mdlm/gosai_data/dataset.csv.gz'), index_col=0) # (735156, 5)
chr_list_1 = [f'chr{i}' for i in range(1, 12)]
chr_list_2 = [f'chr{i}' for i in range(12, 23)]
index_1 = df[df['chrom'].isin(chr_list_1)].index.to_numpy()
index_2 = df[df['chrom'].isin(chr_list_2)].index.to_numpy()

model_params = {
    'model_type':'EnformerPretrainedModel',
    'n_tasks': 3,
}

train_params = {
    'task':'regression',
    'loss': 'MSE',
    'lr':1e-4,
    'logger': 'wandb',
    'batch_size': 512,
    'num_workers': 4,
    'devices': [0],
    'save_dir': os.path.join(base_path, 'mdlm/outputs_gosai'),
    'optimizer': 'adam',
    'max_epochs': 10,
    'checkpoint': True,
}

def train_model(train_index):
    df_train = df.loc[train_index][['seq', 'hepg2', 'k562', 'sknsh']]
    df_val = df.loc[~df.index.isin(train_index)][['seq', 'hepg2', 'k562', 'sknsh']]
    # Make labeled datasets
    train_dataset = grelu.data.dataset.DFSeqDataset(df_train)
    val_dataset   = grelu.data.dataset.DFSeqDataset(df_val)

    model = LightningModel(model_params=model_params, train_params=train_params)
    trainer = model.train_on_dataset(train_dataset, val_dataset)
    wandb.finish()
    return model

model_1 = train_model(index_1)
model_2 = train_model(index_2)
