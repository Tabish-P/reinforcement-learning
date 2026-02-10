# Deep Reinforcement Learning - Value-Based Methods

This repository demonstrates the application of value-based reinforcement learning methods to solve complex control problems. 

### Navigation (Unity ML-Agents)

**Problem**: Train an agent to navigate a 3D environment and collect yellow bananas while avoiding blue bananas.

**Environment Details**:
- **State Space**: 37 dimensions (agent velocity + ray-based perception)
- **Action Space**: 4 discrete actions (forward, backward, left, right)
- **Reward**: +1 for yellow banana, -1 for blue banana
- **Goal**: Achieve average score ≥ 13 over 100 consecutive episodes

**Environment**: Custom Unity ML-Agents environment (Banana.app)

**Key Components**:
- [`dqn_agent_unity.py`](Navigation/dqn_agent_unity.py) - DQN agent for Unity environments
- [`model_unity.py`](Navigation/model_unity.py) - Neural network for navigation task
- [`Navigation.ipynb`](Navigation/Navigation.ipynb) - Interactive training notebook
- [`checkpoint.pth`](Navigation/checkpoint.pth) - Trained model weights

---

## Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# Required packages (see requirements.txt)
pip install -r requirements.txt
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Tabish-P/reinforcement-learning.git
cd reinforcement-learning
```

2. **Install dependencies**:
```bash
pip install torch numpy matplotlib pygame imageio unityagents
```

3. **For Navigation project**: The Unity ML-Agents environment (`Banana.app`) must be downloaded and added to the folder

### Running the Projects

```bash
cd Navigation
jupyter notebook Navigation.ipynb
```

- The notebook will initialize the Banana environment
- Train the DQN agent to navigate and collect bananas
- Results are saved to checkpoint.pth

---

## References

- **Mnih et al. (2015)**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Original DQN paper
- **Unity ML-Agents**: [Official Documentation](https://github.com/Unity-Technologies/ml-agents)
- **PyTorch**: [Official Documentation](https://pytorch.org/)
- **Udacity DRL**: [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

## Author & Attribution

**Implementation**: Tabish Punjani  
**Udacity Project**: Deep Reinforcement Learning Nanodegree  
**Base Code**: Provided by Udacity  

---