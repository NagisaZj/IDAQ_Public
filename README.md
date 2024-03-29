# Offline Meta Reinforcement Learning with In-Distribution Online Adaptation
<!-- 
> Recent offline meta-reinforcement learning (meta-RL) methods typically utilize task-dependent behavior policies (e.g., training RL agents on each individual task) to collect a multi-task dataset. However, these methods always require extra information for fast adaptation, such as offline context for testing tasks. To address this problem, we first formally characterize a unique challenge in offline meta-RL: transition-reward distribution shift between offline datasets and online adaptation. Our theory finds that out-of-distribution adaptation episodes may lead to unreliable policy evaluation and that online adaptation with in-distribution episodes can ensure adaptation performance guarantee. Based on these theoretical insights, we propose a novel adaptation framework, called In-Distribution online Adaptation with uncertainty Quantification (IDAQ), which generates in-distribution context using a given uncertainty quantification and performs effective task belief inference to address new tasks. We find a return-based uncertainty quantification for IDAQ that performs effectively. Experiments show that IDAQ achieves state-of-the-art performance on the Meta-World ML1 benchmark compared to baselines with/without offline adaptation.


## Installation
To install locally, you will need to first install [MuJoCo](https://www.roboti.us/index.html). For task distributions in which the reward function varies (Cheetah, Ant), install MuJoCo150 or plus. Set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) as well as the gpu drivers (something like `/usr/lib/nvidia-390`, you can find your version by running `nvidia-smi`).

For the remaining dependencies, create conda environment by
```
conda env create -f environment.yaml
```
Note that we use the Meta-World version with commit b0b66d1f56bbe6c1a4f3df0eb64a6dc3c38429b3

```
git clone https://github.com/Farama-Foundation/Metaworld.git
cd Metaworld
git reset --hard b0b66d1f56bbe
pip install -e .
```
<!-- For task distributions where the transition function (dynamics)  varies  -->

**For Walker environments**, MuJoCo131 is required.
Simply install it the same way as MuJoCo200. To swtch between different MuJoCo versions:

```
export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mjpro${VERSION_NUM}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro${VERSION_NUM}/bin
``` 




## Data Generation


Example of training policies and generating trajectories on multiple tasks:
For point-robot and cheetah-vel:
```
python policy_train.py ./configs/cpearl-sparse-point-robot.json   # actually dense reward is used. To run the sparse reward version, uncomment line 205 in ./Offline-MetaRL/rlkit/envs/point_robot.py.
python policy_train.py ./configs/cheetah-vel.json
python policy_train_mt1.py ./configs/cpearl-mt1.json
```

For Meta-World ML1 tasks (you can modify the task in `./configs/ml1.json`):
```
python data_collection_ml1.py  ./configs/ml1.json
```

Generated data will be saved in `./data/`

## Offline RL Experiments
To reproduce an Meta-World ML1 experiment, run: 
```
run_ml1.sh
```
To run different tasks, modify "env_name" in `./configs/cpearl-ml1.json` as well as "datadirs" in `run_ml1.sh`.

Similarly, for point-robot and cheetah-vel:
```
run_point.sh
run_cheetah.sh
```
