# Vector Datasets Specification and Generation

This directory documents the raw numerical dataset specifications and mathematical models used for the Distributed Vector Processing evaluation.

## 1. Dataset Generation Model

To benchmark massive workloads ($10^6$ up to $5 \times 10^7$ double-precision elements) without relying on gigabytes of static disk I/O bottlenecks, input vectors $X$ and $Y$ are generated using a deterministic trigonometric series:

$$X[i] = \sin\left((i \pmod{1000}) \times 0.01\right) + 1.5$$
$$Y[i] = \cos\left((i \pmod{1000}) \times 0.01\right) + 2.0$$

### Key Benefits:
- **Reproducibility:** Every execution produces identical deterministic floating-point values across any number of MPI ranks ($P = 1, 2, 4, 8, 16$).
- **Numerical Stability:** Values remain bounded within $[0.5, 2.5]$ for $X$ and $[1.0, 3.0]$ for $Y$, preventing floating-point overflow or underflow during large-scale reductions.
- **Zero Disk Latency:** Allows in-situ domain initialization directly in memory, separating memory bandwidth and compute performance from physical storage bottlenecks.

---

## 2. Benchmark Workload Sizing & Memory Footprint

Each double-precision floating-point number occupies $8\text{ bytes}$.  
For four vectors ($X, Y, Z, W$):

$$\text{Total Memory (MB)} = \frac{4 \times N \times 8}{1024 \times 1024}$$

| Workload Tier | Vector Length ($N$) | Single Vector Size | Working Set ($X, Y, Z, W$) | Computational Intensity |
| :--- | :--- | :--- | :--- | :--- |
| **Small** | $1,000,000$ ($10^6$) | $7.63\text{ MB}$ | **$30.52\text{ MB}$** | L3 Cache Sensitive |
| **Medium** | $10,000,000$ ($10^7$) | $76.29\text{ MB}$ | **$305.18\text{ MB}$** | RAM Bandwidth Bound |
| **Large** | $50,000,000$ ($5 \times 10^7$) | $381.47\text{ MB}$ | **$1,525.88\text{ MB}$** ($1.5\text{ GB}$) | Multi-Core Compute Bound |
| **Stress** | $100,000,000$ ($10^8$) | $762.94\text{ MB}$ | **$3,051.76\text{ MB}$** ($3.05\text{ GB}$) | Massive Parallel Workload |

---

## 3. Running Ground Truth Verification

To run analytical verification using NumPy:

```bash
python3 data/generate_data.py --size 10000000
```
