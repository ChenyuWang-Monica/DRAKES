conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install packaging
pip install ninja
pip install transformers
pip install datasets
pip install omegaconf
conda install ipykernel
python -m ipykernel install --user --name sedd --display-name "Python (sedd)"
pip install hydra-core --upgrade
pip install hydra-submitit-launcher

# for mdlm
pip install causal-conv1d
pip install lightning
pip install timm
pip install rich

pip install scipy
pip install wandb
