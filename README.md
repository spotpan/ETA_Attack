# Physically-Aware Backdoor Attacks on ETA Prediction Models

This repository provides the official implementation of the experiments in the paper:

**Physically-Aware Backdoor Attacks on Trajectory-Based ETA Prediction Models**

The code is based on **TTPNet**, a trajectory-based ETA prediction model, and implements
physically realizable trajectory-level backdoor attacks together with adaptive trigger
optimization and evaluation pipelines.

---

## 1. Repository Structure

```text
TTPNet-master/
├── attacks/
|    ├── backdoor_attack.py
|    └── backdoor_attack_acceleration.py
├── main.py
├── TTPNet.py
├── PredictionBiLSTM.py
├── SpeedRoadLSTM.py
├── Speed.py
├── Road.py
├── Attr.py
├── data_loader.py
├── utils.py
├── logger.py
├── Config/
│   ├── Config_128.json
│   └── embedding_128.npy
├── saved_weights/
├── results/
├── logs/
├── scripts/
    ├── plot_trajectory_comparison.py
    └── plot_trajectory_comp_google_api.py
    └──plot_trajectory_salient_map.py
    └──plot_trajectory_salient_map_inone.py
    └── plot_trajectory_salient_map_six.py
├── trajectory_6panel.png
├── trajectory_map.html
└── README.md
```

---


> **Note**:  
> For reproducibility and repository size constraints, we provide the complete
> implementation for **TTPNet**, while other baseline models are represented by
> minimal attack modules only (e.g., `backdoor_attack.py`).

---

## 2. Environment Setup

We recommend using Python 3.8+ and PyTorch.

```bash
conda create -n eta_atk python=3.8
conda activate eta_atk
pip install torch numpy
```


## 3. Model Overview: TTPNet

TTPNet is a trajectory-based ETA prediction model consisting of:

Road embedding + Road LSTM

Short-term and long-term speed encoders

Bi-directional LSTM for trajectory aggregation

Multi-step ETA prediction head

The complete model is implemented in TTPNet.py and its modular components
(Road.py, Speed.py, PredictionBiLSTM.py, etc.).


## 4. Backdoor Attack Pipeline

All experiments in this repository are executed via:

backdoor_attack.py

train_clean : Train a clean (non-backdoored) model

train_triggered : Train a backdoored model with physically-aware triggers

tune : Adaptive trigger optimization

evaluate : Evaluate clean and triggered trajectories


## 5. Model Checkpoint Naming Convention

All checkpoints are stored under saved_weights/.

5.1 Backdoored Models (Main Results)
triggered_model_adv_<J>_<num>.pth
triggered_model_adv_<J>_<num>_train.pth
triggered_model_adv_<J>_<num>_adpt.pth
triggered_model_adv_<J>_<num>_tune.pth


<P> : trajectory partition / trigger placement strategy
(e.g., J-K, A-G)

<num> : trigger size (number of injected segments)

train : after initial backdoor training

adpt : after adaptive trigger refinement

tune : different adapt strategy


## 6. Recommended Evaluation Commands
6.1 Baseline Evaluation
python backdoor_attack.py --task evaluate \
  --weight_file saved_weights/triggered_model.pth \
  --batch_size 256 \
  --result_file results/eval_baseline.csv \
  --log_file eval_baseline

### 6.2 Main Backdoor Results (Recommended)

Partition Q is recommended for clear visualization and comparison.

# Trigger size = 1
python backdoor_attack.py --task evaluate \
  --weight_file saved_weights/triggered_model_adv_Q_1_tune.pth \
  --batch_size 256 \
  --result_file results/eval_Q_k1.csv \
  --log_file eval_Q_k1

# Trigger size = 3
python backdoor_attack.py --task evaluate \
  --weight_file saved_weights/triggered_model_adv_Q_3_tune.pth \
  --batch_size 256 \
  --result_file results/eval_Q_k3.csv \
  --log_file eval_Q_k3

## 7. Visualization Utilities

The following scripts are provided for trajectory and trigger visualization:

plot_trajectory_comp_google_api.py

plot_trajectory_salient_map.py

plot_trajectory_salient_map_six.py

These scripts generate figures used in the paper for illustrating
physically-aware trajectory perturbations.
