## Regulatory DNA Sequence Design
This codebase is developed on top of [MDLM (Sahoo et.al, 2023)](https://github.com/kuleshov-group/mdlm).

### Environment Installation
```
conda create -n sedd python=3.9.18
conda activate sedd
bash env.sh

# install grelu
git clone https://github.com/Genentech/gReLU.git
cd gReLU
pip install .
```

### Data and Model Weights
All data and model weights can be downloaded from this link:

https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&st=xe8dzn8k&dl=0

Save the downloaded file in `BASE_PATH`.

### Gosai Dataset
The enhancer dataset used for this experiment is provided in `BASE_PATH/mdlm/gosai_data`.

### Pretrained Generative Model
```
python main_gosai.py
```
The pretrained model weights are provided in `BASE_PATH/mdlm/outputs_gosai/pretrained.ckpt`.

### Reward Oracle
```
python train_oracle.py
```
The oracle for fine-tuning is provided in `BASE_PATH/mdlm/outputs_gosai/lightning_logs/reward_oracle_ft.ckpt`; the oracle for evaluation is provided in `BASE_PATH/mdlm/outputs_gosai/lightning_logs/reward_oracle_eval.ckpt`.

The oracle for binary classification on chromatin accessibility (ATAC-Acc) is provided in `BASE_PATH/mdlm/gosai_data/binary_atac_cell_lines.ckpt`.


### Fine-Tune to Optimize Enhancer Activity
```
python finetune_reward_bp.py --name test
```
The fine-tuned model weights are provided in `BASE_PATH/mdlm/reward_bp_results_final/finetuned.ckpt`

### Evaluation
See `eval.ipynb`

### Adaptations
Change the `base_path` in `dataloader_gosai.py`, `finetune_reward_bp.py`, `oracle.py`, `train_oracle.py`, `eval.ipynb` to `BASE_PATH` for saving data and models.

### Acknowledgement 

* The original dataset is provided by [Gosai et al., 2023](https://www.biorxiv.org/content/10.1101/2023.08.08.552077v1).
* The trained oracle is based on [gReLU](https://genentech.github.io/gReLU/). 