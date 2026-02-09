# ChronoPlastic Spiking Neural Networks

**Adaptive Time Warping for Long Horizon Memory**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Under Review at ICML 2026**

## Abstract

Spiking neural networks (SNNs) offer a biologically grounded and energy-efficient alternative to conventional neural architectures; however, they struggle with long-range temporal dependencies due to fixed synaptic and membrane time constants. This paper introduces **ChronoPlastic Spiking Neural Networks (CPSNNs)**, a novel architectural principle that enables adaptive temporal credit assignment by dynamically modulating synaptic decay rates conditioned on the state of the network.

CPSNNs maintain multiple internal temporal traces and learn a continuous time-warping function that selectively preserves task-relevant information while rapidly forgetting noise. Unlike prior approaches based on adaptive membrane constants, attention mechanisms, or external memory, CPSNNs embed temporal control directly within local synaptic dynamics, preserving linear-time complexity and neuromorphic compatibility.

### Notable Results

| Model | Accuracy (Large Gap) |
|-------|---------------------|
| Standard SNN | 0.52 |
| Adaptive Membrane SNN | 0.61 |
| **CPSNN (Ours)** | **0.98** |

- Up to 40% absolute accuracy gains on long-gap benchmarks
- 98% accuracy maintained at gap lengths where conventional SNNs degrade severely
- Almost 3× faster convergence without increasing model complexity

## Method Overview

### The ChronoPlastic Synapse

Each ChronoPlastic synapse maintains two internal temporal traces:

```
fₜ = αf · fₜ₋₁ + sₜ           (fast trace)
zₜ = αs^ωₜ · zₜ₋₁ + sₜ        (slow trace with adaptive decay)
```

The adaptive warp factor `ωₜ ∈ (0,1)` is produced by a lightweight control network:

```
ωₜ = σ(g([sₜ, zₜ₋₁]))
```

This formulation dynamically rescales time itself—smaller `ωₜ` slows decay and extends memory, while larger values accelerate forgetting.

### Synaptic Current

```
Iₜ = W·sₜ + λf·W·fₜ + λs·W·zₜ
```

This decomposition enables representation across a continuum of timescales without explicit memory storage or attention mechanisms.

## Installation

```bash
git clone https://github.com/Sarimsaljook/chronoplastic_snn.git
cd chronoplastic_snn
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy
- Matplotlib

## Quick Start

### Basic Demo

```bash
python demo.py
```

Trains a CPSNN on the long-gap temporal XOR task and visualizes the learning curve.

### Full Experiment

```bash
python main.py --epochs 40 --hidden 256 --lr 0.01
```

Compares CPSNN against baseline SNNs with progressively increasing temporal gaps.

### Fast Divergence Analysis

```bash
python fast-divergence.py --epochs 40 --hidden 256 --lr 0.01
```

Demonstrates the performance gap between CPSNN and standard SNNs under challenging conditions.

## Repository Structure

```
├── demo.py                 # Working long-gap example
├── main.py                 # Complete training pipeline with comparisons
├── fast-divergence.py      # Stress test with increasing gaps
├── normalSNN_Demo.py       # Baseline SNN implementation
├── requirements.txt        # Dependencies
└── README.md
```

## Usage

### Core Components

**ChronoPlastic Synapse**
```python
from main import ChronoPlasticSynapse

synapse = ChronoPlasticSynapse(
    c_in=16,           # Input channels
    c_out=256,         # Output channels
    alpha_fast=0.90,   # Fast trace decay
    alpha_slow=0.995   # Slow trace base decay
)
```

**LIF Layer with ChronoPlastic Synapses**
```python
from main import LIFLayer, LIFParams

lif_params = LIFParams(v_th=1.0, v_reset=0.0, dt=1e-3, tau_mem=2e-2)
layer = LIFLayer(c_in=16, c_out=256, lif=lif_params, chrono=True)
```

**Full Network**
```python
from main import SpikingNet, LIFParams

model = SpikingNet(
    C_in=16,
    C_hidden=256,
    C_out=256,
    lif=LIFParams(),
    chrono=True  # Enable ChronoPlastic synapses
)
```

### Training

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    for X, y in dataloader:
        X = X.transpose(0, 1)  # [T, B, C]
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_fast` | 0.90 | Fast trace decay constant |
| `alpha_slow` | 0.995 | Slow trace base decay constant |
| `λf` | 0.5 | Fast trace mixing coefficient |
| `λs` | 0.5 | Slow trace mixing coefficient |
| `learning_rate` | 0.01 | Adam optimizer learning rate |
| `grad_clip` | 1.0 | Gradient clipping norm |

## Computational Complexity

| Aspect | CPSNN | Attention-based |
|--------|-------|-----------------|
| Time | O(T·N·C) | O(T²·N) |
| Space | O(N·C) | O(T·N) |

## Acknowledgments

This work builds upon foundational research in spiking neural networks, surrogate gradient methods, and neuromorphic computing. We thank the broader SNN community for their contributions to trainable spiking systems.
