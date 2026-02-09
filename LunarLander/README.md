# Deep Q-Network (DQN) for LunarLander-v3

## Project Overview

This project implements a **Deep Q-Network (DQN)** agent to solve the OpenAI Gymnasium **LunarLander-v3** environment. It's part of the **Udacity Deep Reinforcement Learning Nanodegree** program.

The project has been enhanced from its original state:
- ✅ Upgraded to LunarLander-v3 (latest version with improved physics)
- ✅ Integrated **MPS** backend for Apple Silicon M3 acceleration
- ✅ Successfully trained and tested the agent
- 🔄 Planning: Hyperparameter optimization and advanced RL techniques

---

## Current Status

### ✅ Achievements

| Metric | Value |
|--------|-------|
| **Environment Solved** | ✓ Yes |
| **Episodes to Solve** | 1,711 |
| **Final Average Score** | 260.40 |
| **Training Time** | ~22 minutes (with MPS acceleration) |
| **Device** | Apple M3 MacBook Air (GPU accelerated) |
| **Checkpoint Saved** | checkpoint.pth |

### Performance Curve

The agent exhibited smooth learning progression:
- **Episodes 1-200**: Exploration phase (negative rewards ~-150)
- **Episodes 200-500**: Initial learning breakthrough
- **Episodes 500-1000**: Steady improvement (200+ scores)
- **Episodes 1000-1711**: Convergence to solution (260+ target)

---

## Technical Setup

### Requirements

```
Python 3.14+
gymnasium >= 1.2.0
torch >= 2.10.0
numpy >= 2.4.0
matplotlib >= 3.8.0
box2d-py >= 2.3.0
pygame >= 2.6.0
imageio >= 2.0.0
```

### Installation

1. **Clone the repository**
```bash
cd ~/Udacity/reinforcement-learning/LunarLander
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install gymnasium torch numpy matplotlib box2d-py pygame imageio
```

4. **For M3/M4 MacBook users**: MPS is automatically detected and used if available
```python
# Verify MPS availability
import torch
print(torch.backends.mps.is_available())  # Should return True
```

---

## DQN Algorithm Overview

### Core Concept

The agent learns a **Q-function**: $Q(s, a) \approx$ expected future reward for action *a* in state *s*

### Key Mechanisms

1. **Two Networks**
   - Local network: Updated frequently
   - Target network: Updated slowly (soft update with τ = 0.001)
   - Provides stability during learning

2. **Experience Replay**
   - Stores experiences in buffer (100k capacity)
   - Randomly samples batches (64 samples)
   - Breaks temporal correlations

3. **Q-Learning Update**
   - Target: $Q_{target} = r + \gamma \max Q(s', a')$
   - Loss: MSE between target and predicted Q-value
   - Optimizer: Adam (lr = 0.0005)

4. **Exploration Strategy**
   - ε-greedy: Random action with probability ε
   - ε decays from 1.0 → 0.01 over training
   - Shift from exploration → exploitation

---

### Comparison to Expectations

- **LunarLander-v2**: Typically solves in 1000-1500 episodes
- **LunarLander-v3**: Higher bar (score ≥ 260 vs 200), solved in 1711 episodes
- **With MPS**: Significant wall-clock speedup while maintaining same episode count

---

## 🔄 Future Improvements

### 1. **Hyperparameter Optimization**

Test variations:
```python
# Exploration decay
eps_decay: [0.99, 0.985, 0.975, 0.96]

# Learning rate
LR: [1e-4, 5e-4, 1e-3, 2e-3]

# Target network update
TAU: [1e-3, 5e-3, 1e-2, 5e-2]

# Batch size
BATCH_SIZE: [32, 64, 128, 256]
```

**Goal**: Achieve convergence in < 1000 episodes

### 2. **Advanced RL Techniques**

- [ ] **Double DQN**: Use target network for action selection (reduces overestimation)
- [ ] **Dueling DQN**: Separate state value and advantage streams
- [ ] **Prioritized Experience Replay**: Oversample important experiences
- [ ] **Rainbow DQN**: Combine multiple improvements

### 3. **Network Architecture Experiments**

- [ ] Wider networks (128 neurons per layer)
- [ ] Deeper networks (3-4 hidden layers)
- [ ] Batch normalization for training stability
- [ ] Dueling architecture (value + advantage heads)

## References

### Foundational Papers
- Mnih et al. (2015): "[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)"

### Udacity Resources
- [Deep RL Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
- [Project Specifications](https://github.com/udacity/deep-reinforcement-learning)

### Library Documentation
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch MPS Backend](https://pytorch.org/docs/master/notes/mps.html)
- [PyTorch Lightning Documentation](https://www.pytorchlightning.ai/)

---

## Author & Attribution

**Implementation**: Tabish Punjani  
**Udacity Project**: Deep Reinforcement Learning Nanodegree  
**Base Code**: Provided by Udacity  

---