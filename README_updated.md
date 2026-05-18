# R-Sync: Reward-Synchronized Decentralized Federated Learning

[![IEEE IoT Journal](https://img.shields.io/badge/IEEE-IoT%20Journal-blue)](https://ieee-iotj.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![CARLA 0.9.16](https://img.shields.io/badge/CARLA-0.9.16-orange)](https://carla.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red)](https://pytorch.org)

> **Reward-Synchronized DFL Based on SARSA for Autonomous Physical AI Networks**
> IEEE Internet of Things Journal (Under Review)

---

## Overview

R-Sync is a decentralized federated learning framework for Physical AI networks in which each node independently maintains an on-policy **SARSA** agent that learns an adaptive synchronization participation policy. The participation decision is governed by a three-component sigmoid gate:

$$P(\text{participate}) = \sigma\bigl(\alpha \cdot \text{acc} + \beta \cdot w_i - \gamma \cdot \hat{s}_i \cdot \theta\bigr)$$

| Symbol | Meaning |
|--------|---------|
| `acc` | Local model accuracy (previous round R²) |
| `w_i` | Distribution-awareness weight (local std / max std) |
| `ŝ_i` | Estimated straggler score |
| `θ` | SARSA action (participation aggressiveness) |
| `α, β, γ` | Weighting coefficients |

Non-participating nodes broadcast their barrier signal **before** training begins, immediately releasing the synchronization barrier without idle overhead.

---

## Key Results

| Method | Avg. Waiting Time (s) | Avg. Test R² |
|--------|-----------------------|--------------|
| **R-Sync (proposed)** | **0.1757** | **0.8591** |
| OORT | 0.3980 | 0.8229 |
| FedCS | 0.3971 | 0.8067 |
| FedBuff | 0.4113 | 0.8046 |

- **67.3%** waiting time reduction vs BSP (FedAvg)
- **48.8% – 76.4%** bandwidth saving across K ∈ {3,...,7}

---

## Hardware

- 7 × Raspberry Pi 5 (8 GB RAM, quad-core Cortex-A76)
- Fully connected peer-to-peer HTTP topology
- No central parameter server

---

## Dataset

Multi-modal autonomous vehicle speed prediction dataset generated using **CARLA 0.9.16**.

| Split | Towns | Samples per node |
|-------|-------|-----------------|
| Training | Town01–06, Town10HD | 1,233 – 3,136 |
| Test (held-out) | Town07 | 5,000 |

**20-dimensional feature vector:** steering, throttle, brake, LiDAR/radar distance, GPS position, IMU (yaw/pitch/roll, acceleration, gyroscope), LiDAR mean distance, radar mean distance/velocity, mean depth.

**Target:** `speed_kmh` (continuous regression)

---

## Repository Structure

```
.
├── collect_data.py              # CARLA data collection (Step 1)
├── create_noniid_dataset.py     # Dirichlet non-IID partitioning (Step 2)
├── client.py                    # R-Sync main client (SARSA)
├── network/
│   ├── http_server.py           # Per-node HTTP server
│   └── http_client.py           # HTTP client for peer communication
├── utils/
│   ├── model_utils.py           # MLP model, training, evaluation
│   └── sync_utils.py            # SARSA utilities
├── baselines/
│   ├── fedcs_client.py          # FedCS baseline
│   ├── fedbuff_client.py        # FedBuff baseline
│   └── oort_client.py           # OORT-inspired baseline
├── config/
│   └── peer_config.yaml         # Node IPs, ports, hyperparameters
└── data/                        # Generated datasets (not tracked)
    ├── Town01/
    ├── Town02/
    ...
    └── Town07/                  # Held-out test set
```

---

## Quick Start

### Step 1 — Collect Raw Data (CARLA required)

```bash
# Training towns
python collect_data.py --town Town01
python collect_data.py --town Town02
python collect_data.py --town Town03
python collect_data.py --town Town04
python collect_data.py --town Town05
python collect_data.py --town Town06
python collect_data.py --town Town10HD

# Held-out test set (no Dirichlet partitioning applied)
python collect_data.py --town Town07
```

Each run collects **5,000 frames** and saves `data/{TOWN}/driving_log.csv`.

### Step 2 — Apply Dirichlet Non-IID Partitioning

```bash
python create_noniid_dataset.py
```

Output: `client_X_nonIID_alpha0.5_nYYYY.csv` for each training node.

### Step 3 — Configure Nodes

Edit `config/peer_config.yaml`:

```yaml
peers:
  1:
    ip: 192.168.0.2
    port: 9000
    dataset_path: /path/to/client_1_nonIID_alpha0.5_n1233.csv
  ...
```

### Step 4 — Run R-Sync

On each Raspberry Pi node (run simultaneously):

```bash
# Node 1
python client.py 1

# Node 2
python client.py 2

# ... up to Node 7
python client.py 7
```

### Step 5 — Run Baselines (optional)

```bash
# FedCS
python baselines/fedcs_client.py 1   # on each node

# FedBuff
python baselines/fedbuff_client.py 1

# OORT-inspired
python baselines/oort_client.py 1
```

---

## Model Architecture

Two-hidden-layer MLP with dropout:

```
Input (20) → Linear(128) → ReLU → Dropout(0.2)
           → Linear(64)  → ReLU → Dropout(0.2)
           → Linear(1)   → output (speed_kmh)

Total parameters: 11,009
Model size: ~43 KB (float32)
```

---

## SARSA Configuration (WaitWeight — recommended)

```yaml
sarsa:
  lambda1: 1.8    # α — accuracy weight
  lambda2: 1.5    # β — distribution weight
  lambda3: 4.0    # γ — straggler penalty (WaitWeight)
  learning_rate: 0.12
  gamma: 0.92
  epsilon: 0.15
```

---

## Requirements

```
Python >= 3.8
torch >= 1.13
numpy
pandas
scikit-learn
scipy
flask
aiohttp
pyyaml
tqdm
opencv-python   # for data collection only
carla == 0.9.16 # for data collection only
```

Install:

```bash
pip install torch numpy pandas scikit-learn scipy flask aiohttp pyyaml tqdm
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{rsync2025,
  title   = {Reward-Synchronized DFL Based on SARSA for
             Autonomous Physical AI Networks},
  journal = {IEEE Internet of Things Journal},
  year    = {2025},
  note    = {Under Review}
}
```

---

## License

This project is released for academic research purposes.
