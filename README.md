# A Walk in the Park

Code to replicate [A Walk in the Park: Learning to Walk in 20 Minutes With Model-Free Reinforcement Learning](https://arxiv.org/abs/2208.07860), which contains code for training a simulated or real A1 quadrupedal robot to walk. Project page: https://sites.google.com/berkeley.edu/walk-in-the-park

## Installation step [Josselin]
Install [MujoCo](https://github.com/openai/mujoco-py#install-mujoco).
The version `mjpro150` is required, you can find it [here](https://www.roboti.us/download.html). Place it at this path: `~/mujoco/mjpro150`.

Then download a key for Mujoco [here](https://www.roboti.us/license.html) and place it in `~/.mujoco/mjkey.txt`.

Add the following line to `~/.bashrc`:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/josselin/.mujoco/mjpro150/bin
```
An then refresh your terminal with:
```bash
source ~./bashrc
```

Create venv with:
```bash
python -m venv
```

Activate venv with:
```bash
source venv/bin/activate
```

Download a few libraries: (as described [here](https://github.com/openai/mujoco-py/))
```bash
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Install `patchelf` to solve this [issue](https://github.com/openai/mujoco-py/issues/652):
```bash
sudo apt-get install patchelf
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a [wandb](https://wandb.ai) account, join the [single-shot-robot](https://wandb.ai/single-shot-robot) project or create your own and copy your API key somewhere you can access later.


Run the simulation:
```bash
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=A1Run-v0 \
                --utd_ratio=20 \
                --start_training=1000 \
                --max_steps=100000 \
                --config=configs/droq_config.py
```

When asked how to run `wandb` type: `2` for `Use an existing W&B account` and then enter your API key.

***I have not tried to build the SDK, I only work with the simulation for now.***

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

To install the robot [SDK](https://github.com/unitreerobotics/unitree_legged_sdk), first install the dependencies in the README.md

To build, run: 
```bash
cd real/third_party/unitree_legged_sdk
mkdir build
cd build
cmake ..
make
``` 

Finally, copy the built `robot_interface.XXX.so` file to this directory.

## Training

Example command to run simulated training:

```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=A1Run-v0 \
                --utd_ratio=20 \
                --start_training=1000 \
                --max_steps=100000 \
                --config=configs/droq_config.py
```

To run training on the real robot, add `--real_robot=True`

