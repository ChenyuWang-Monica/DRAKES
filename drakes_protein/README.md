## Protein Sequence Design: Optimizing Stability in Inverse Folding Model
This codebase is developed on top of [MultiFlow (Campbell & Yim et.al, 2024)](https://github.com/jasonkyuyim/multiflow).

### Environment Installation
For the environment installation, please refer to [MultiFlow](https://github.com/jasonkyuyim/multiflow) for details.
```
# Install environment with dependencies.
conda env create -f multiflow.yml

# Activate environment
conda activate multiflow

# Install local package.
# Current directory should have setup.py.
pip install -e .

# Install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

Then install [PyRosetta](https://www.pyrosetta.org/downloads).
```
pip install pyrosetta-installer 
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

### Data and Model Weights
All data and model weights can be downloaded from this link:

https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&st=xe8dzn8k&dl=0

Save the downloaded file in `BASE_PATH`.

### Pretrained Inverse Folding Model
We use the PDB dataset utilized in [ProteinMPNN](https://github.com/dauparas/ProteinMPNN/tree/main/training) for the inverse folding model pretraining. The multi-chain training data (16.5 GB, PDB biounits, 2021 August 2) can be downloaded from here: `https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz`. Svae the downloaded file in `BASE_PATH/pmpnn/raw`.

```
# pretrain the model
python train_fmif.py --eval_every_n_epochs 100
```

The pretrained model weights are provided in `BASE_PATH/pmpnn/outputs/pretrained_if_model.pt`.

### Megascale Dataset
We use the [Megascale](https://www.nature.com/articles/s41586-023-06328-6) dataset, following the preprocessing by [ProteinDPO](https://github.com/evo-design/protein-dpo), for training the reward oracles and fine-tuning the inverse folding model. The processed data is in `BASE_PATH/proteindpo_data`.


### Reward Oracle
#### Oracle for Fine-Tuning
```
python train_oracle.py --save_model_every_n_epochs=5 --wandb_name=test_ft
```
The oracle is provided in `BASE_PATH/protein_oracle/outputs/reward_oracle_ft.pt`.

#### Oracle for Evaluation
```
python train_oracle.py --save_model_every_n_epochs=5 --wandb_name=test_eval --include_all=True --num_epochs=100
```
The oracle is provided in `BASE_PATH/protein_oracle/outputs/reward_oracle_eval.pt`.


### Fine-Tune to Optimize Protein Stability
```
python finetune_reward_bp.py --wandb_name=test
```
The fine-tuned model weights are provided in `BASE_PATH/protein_rewardbp/finetuned_if_model.ckpt`


### Evaluation
```
cd scripts/
bash ours.sh
```

### Adaptations
Change the `base_path` in `fmif/finetune_reward_bp.py`, `fmif/eval_finetune.py`, `fmif/train_fmif.py`, `protein_oracle/train_oracle.py` to `BASE_PATH` for saving data and models.

