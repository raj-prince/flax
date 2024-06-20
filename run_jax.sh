#!/bin/bash

# Manual
# git clone https://github.com/raj-prince/flax.git
# cd flax
# git checkout princer_jax

# assuming above commands are run
conda create -yn flax_env python==3.10
conda activate flax_env

pip install -U pip
pip install -e .

cd examples/imagenet
rm -rf ws
mkdir ws

pip install -r requirements.txt
pip install --no-cache-dir tensorrt==8.6.1

ln -s /opt/conda/envs/flax_env/lib/python3.10/site-packages/tensorrt_libs/libnvinfer.so.8 /opt/conda/envs/flax_env/lib/python3.10/site-packages/tensorrt_libs/libnvinfer.so.7
ln -s /opt/conda/envs/flax_env/lib/python3.10/site-packages/tensorrt_libs/libnvinfer_plugin.so.8 /opt/conda/envs/flax_env/lib/python3.10/site-packages/tensorrt_libs/libnvinfer_plugin.so.7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/flax_env/lib/python3.10/site-packages/tensorrt_libs/
export TFDS_DATA_DIR=/mounted-dir-path #set this variable to directory mounted gcsfuse

# Run the model
python main.py --workdir="./ws" --config="./configs/v100_x8.py"
