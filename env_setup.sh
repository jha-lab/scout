#!/bin/sh

# Create a new environment with Python 3.9
conda create -y --name scout python=3.9
conda activate scout
# install transformers
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install --no-cache-dir -e ./transformers
# install other dependencies
pip3 install tqdm ipywidgets matplotlib scikit-optimize
jupyter nbextension enable --py widgetsnbextension
pip3 install pyyaml 