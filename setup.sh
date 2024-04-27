#!/bin/bash

conda activate ctm

apt update
apt upgrade -y

apt install -y gcc
apt install libstdc++6
apt upgrade -y

apt install -y vim git
apt install -y libgl1-mesa-glx
apt install -y libglib2.0-0 libsm6 libxrender1 libxext6
apt install -y tmux
apt install -y wget
apt install -y htop

conda update -n bash conda
conda install -n bash conda-libmamba-solver

conda update -n ctm conda
conda install -n ctm conda-libmamba-solver
conda config --set solver libmamba

conda install -c conda-forge

apt install -y git
apt install -y libopenmpi-dev

python -m pip install --upgrade pip

python -m pip install --upgrade tensorflow[and-cuda]
python -m pip install torch torchvision torchaudio
# python -m pip install torch==2.2.0+cu118 torchaudio==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install blobfile tqdm numpy scipy pandas Cython piq==0.7.0
python -m pip install joblib==0.14.0 albumentations==0.4.3 lmdb clip@git+https://github.com/openai/CLIP.git pillow
python -m pip install flash-attn --no-build-isolation
python -m pip install mpi4py
python -m pip install nvidia-ml-py3 timm==0.4.12 legacy dill nvidia-ml-py3
python -m pip install chardet pytorch-msssim lpips munch
