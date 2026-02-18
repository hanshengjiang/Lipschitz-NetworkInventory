# Learning in Continuous State-Space MDPs for Network Inventory Management

[![AISTATS 2026](https://img.shields.io/badge/AISTATS-2026-blue.svg)](https://openreview.net/forum?id=e4ATBG2Oh5)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors:** [Hansheng Jiang](https://openreview.net/profile?id=~Hansheng_Jiang1), [Shunan Jiang](https://openreview.net/profile?id=~Shunan_Jiang1), [Zuo-Jun Shen](https://openreview.net/profile?id=~Zuo-Jun_Shen1)

**Paper:** [OpenReview](https://openreview.net/forum?id=e4ATBG2Oh5)

---

## Overview

This repository contains the implementation of **LipBR** (Lipschitz Bandits-based Repositioning), an online learning algorithm for managing multi-location inventory networks under unknown demand and censored sales data. Our work addresses the challenging problem of learning in infinite-horizon, average-cost Markov Decision Processes (MDPs) with multi-dimensional, continuous state spaces.

### Key Contributions

- **Novel Framework:** We establish the Lipschitz property of the long-run average cost function in network inventory systems, enabling analysis through the lens of Lipschitz bandits.

- **Provably Efficient Algorithm:** LipBR achieves a high-probability regret bound of $\tilde{O}(T^{\frac{n}{n+1}})$, where $n$ is the network size and $T$ is the time horizon.

- **Matching Lower Bound:** We derive a matching lower bound that captures the inherent dimensionality challenge of the problem.

- **Practical Performance:** Numerical experiments demonstrate that LipBR significantly outperforms baseline policies across various network configurations.

---

## Repository Structure

```
Lipschitz-NetworkInventory/
├── README.md                          # This file
├── LICENSE                            # License information
├── LipschitzMDP_Numerical.ipynb       # Main implementation and experiments
└── LipschitzMDP_AISTATS_CameraReady.pdf  # Camera-ready paper (if available)
```

---

## Installation

### Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Pandas
- Jupyter Notebook

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hanshengjiang/Lipschitz-NetworkInventory.git
cd Lipschitz-NetworkInventory
```

2. Install dependencies:
```bash
pip install numpy matplotlib pandas jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook LipschitzMDP_Numerical.ipynb
```

---

## Usage

### Running Experiments

The main notebook `LipschitzMDP_Numerical.ipynb` contains:

1. **Algorithm Implementation:**
   - `LipBR`: The main LipBR algorithm with UCB-based exploration
   - `NetworkInventoryEnv`: The network inventory environment simulator
   - `simplex_grid`: Discretization of the policy space on the simplex

2. **Baseline Policies:**
   - `simulate_no_reposition`: No repositioning baseline (y_t = x_t)
   - `simulate_uniform_reposition`: Uniform repositioning baseline (y_t = uniform)

3. **Experiments:**
   - Multi-horizon evaluation (T = 1000, 2000, 3000)
   - Multi-network-size experiments (n = 2, 3, 4)
   - Performance comparison across policies
   - Regret analysis

### Quick Start Example

```python
import numpy as np
from dataclasses import dataclass

# Define demand and routing samplers
def demand_sampler(rng, n):
    demand_mu = np.linspace(0.2, 0.8, n)
    return rng.poisson(lam=demand_mu)

def routing_sampler(rng, n):
    return rng.dirichlet(alpha=np.ones(n), size=n)

# Create environment
env = NetworkInventoryEnv(
    n=3,  # 3 locations
    demand_sampler=demand_sampler,
    routing_sampler=routing_sampler,
    cost_reposition=1.0,
    cost_lost=10.0,
)

# Configure and run LipBR
config = LipBRConfig(horizon=1000, grid_m=4, H=5.0)
agent = LipBR(env, config)
history = agent.run(verbose=True)

# Access results
avg_cost = history["avg_modified_cost"][-1]
print(f"Average modified cost: {avg_cost:.3f}")
```

### Customization

You can customize the experiments by modifying:

- **Network size (`n_list`)**: Number of locations in the inventory network
- **Time horizons (`T_list`)**: Length of the learning episodes
- **Cost parameters**: `COST_REPO` (repositioning cost), `COST_LOST` (lost-sales cost)
- **Demand distribution**: Modify `build_demand_mu()` or provide custom samplers
- **Grid resolution**: Adjusted automatically via `recommended_grid_m()`, or set manually

---

## Algorithm Details

### LipBR Algorithm

The LipBR algorithm operates in epochs with doubling lengths:

1. **Initialization:** Discretize the policy space into K arms on the (n-1)-simplex
2. **Epoch-based UCB:** Select arm k with highest UCB value
3. **Consecutive play:** Play arm k for N_k consecutive periods
4. **Pseudo-cost adjustment:** Adjust for state mismatch using memory points
5. **Update:** Update empirical mean and UCB values; double epoch length

**Key Features:**
- Handles continuous state spaces via discretization
- Uses pseudo-costs to account for state-dependent transitions
- Achieves near-optimal regret through adaptive exploration

### Mathematical Formulation

- **State space:** Probability simplex $\Delta_{n-1}$
- **Action space:** Base-stock vectors in $\Delta_{n-1}$
- **Modified cost:** $\tilde{C}_t = M(y_t - x_t) - \ell \sum_i \min(d_{t,i}, y_{t,i})$
- **Regret bound:** $\tilde{O}(T^{\frac{n}{n+1}} (\log T)^{\frac{1}{n+1}})$

---

## Experimental Results

The notebook generates three types of performance tables:

1. **Average Total Cost:** Compares LipBR, NoRepo, and Uniform policies across horizons
2. **Pseudo-regret vs Last Arm:** Reports LipBR's final cumulative pseudo-regret
3. **Average Regret vs LipBR Last Arm:** Benchmarks all policies against LipBR's converged policy

Results are saved in `numerical_results/` with timestamped subdirectories containing:
- CSV files with per-period statistics
- Metadata files (`params.md`) with experiment configuration

---

## Paper Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{jiang2026learning,
  title={Learning in Continuous State-Space MDPs for Network Inventory Management},
  author={Jiang, Hansheng and Jiang, Shunan and Shen, Zuo-Jun},
  booktitle={Proceedings of The 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026},
  url={https://openreview.net/forum?id=e4ATBG2Oh5}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please contact:
- Hansheng Jiang: [hanshengjiang@rotman.utoronto.ca](mailto:hansheng.jiang@rotman.utoronto.ca)

Or open an issue on the [GitHub repository](https://github.com/hanshengjiang/Lipschitz-NetworkInventory).

---

## Acknowledgments

This work was presented at the 29th International Conference on Artificial Intelligence and Statistics (AISTATS 2026). We thank the reviewers for their valuable feedback.
