# Distributed Vector Processing (MPI) — Technical Viva & Defense Guide

**Course:** Parallel and GPU Computing (PGC Lab)  
**Team Assignment:** Topic 5 — Distributed Vector Processing (MPI)  
**Student:** Akash TD (`01FE24BCI081`)

---

## 1. Core Concepts & Architecture

### Q1: What is the fundamental difference between the Distributed-Memory (MPI) and Shared-Memory (OpenMP/Pthreads) models?
- **Shared-Memory (OpenMP/Pthreads):** All threads execute within a single address space and communicate implicitly by reading/writing shared memory locations. Synchronization is required via locks, mutexes, or atomic operations to prevent data races. Scalability is strictly constrained by the physical RAM and core count of a single motherboard (SMP/NUMA node).
- **Distributed-Memory (MPI):** Each process (rank) has its own private, isolated virtual address space. Processes cannot access each other's memory directly. All coordination and data sharing must occur explicitly through message passing over network interfaces or IPC (inter-process communication) sockets. MPI can scale across thousands of distinct compute nodes in a supercomputer cluster.

### Q2: What is 1D Domain Decomposition and how did you partition the vectors?
- For a vector of length $N$ and $P$ MPI processes, the domain is partitioned into contiguous blocks (block distribution).
- **Base chunk size:** $n_{\text{base}} = \lfloor N / P \rfloor$
- **Remainder handling:** If $N \pmod P \neq 0$, the remaining $R = N \pmod P$ elements are distributed one-by-one to the first $R$ ranks. Rank $r$ receives:
  $$\text{local\_n} = \begin{cases} n_{\text{base}} + 1, & \text{if } r < R \\ n_{\text{base}}, & \text{otherwise} \end{cases}$$
- **Displacement Calculation:** `displs[r]` is calculated as the cumulative sum $\sum_{i=0}^{r-1} \text{sendcounts}[i]$. This guarantees zero overlap and contiguous global memory coverage.

### Q3: Why did you use `MPI_Scatterv` and `MPI_Gatherv` instead of `MPI_Scatter` and `MPI_Gather`?
- Standard `MPI_Scatter` and `MPI_Gather` mandate that every process receives and sends the exact same number of elements ($N$ must be strictly divisible by $P$).
- The vector versions (`MPI_Scatterv` and `MPI_Gatherv`) allow variable chunk lengths (`sendcounts` array) and explicit starting offsets (`displs` array), making the code robust and production-ready for **any arbitrary vector size $N$** and any arbitrary process count $P$.

---

## 2. Communication Primitives & Complexity

### Q4: Explain the difference between Collective and Point-to-Point Communication.
- **Point-to-Point (`MPI_Send`, `MPI_Recv`):** Involves exactly two specific processes (sender and receiver). Can be blocking or non-blocking (`MPI_Isend`, `MPI_Irecv`).
- **Collective Communication (`MPI_Bcast`, `MPI_Scatterv`, `MPI_Gather`, `MPI_Reduce`):** Involves all processes in an `MPI_Comm` communicator simultaneously. Collectives provide optimized hardware-level communication topologies (e.g. binomial trees, hypercubes, ring algorithms) that vastly outperform naive loops of point-to-point sends.

### Q5: How does `MPI_Reduce` work internally and what is its time complexity?
- If process 0 simply collected all partial sums using sequential `MPI_Recv`, the time complexity would be $O(P)$ and process 0 would become a serialized bottleneck.
- `MPI_Reduce` utilizes a **binomial tree** or **recursive doubling** reduction tree:
  - Step 1: $P/2$ processes send their values to the other $P/2$ processes, which add them.
  - Step 2: Half of the remaining processes send to the other half.
  - Total communication steps: $\lceil \log_2 P \rceil$.
- Thus, the communication complexity is $O(\log_2 P)$, offering high scalability as $P$ grows.

### Q6: What is the difference between `MPI_Reduce` and `MPI_Allreduce`?
- `MPI_Reduce` aggregates data from all processes and delivers the final reduced result only to the designated `root` process (Rank 0).
- `MPI_Allreduce` performs the same reduction, but distributes the final reduced scalar/vector to **every process** in the communicator (equivalent to `MPI_Reduce` followed by `MPI_Bcast`).

---

## 3. Performance, Scalability & Amdahl's Law

### Q7: What are the two workflow models evaluated in this experiment?
1. **Centralized Master-Worker (Scatter-Gather):**
   - Rank 0 allocates full vectors $X, Y$, scatters them via `MPI_Scatterv`, processes calculate locally, and results $Z, W$ are collected back via `MPI_Gatherv`.
   - Communication complexity is $O(N)$ doubles across the memory bus.
2. **In-Situ Distributed Domain Decomposition:**
   - Each rank initializes and processes its assigned slice directly in local memory.
   - Only scalar results (dot product, norm, sum) are reduced via `MPI_Reduce` ($O(\log P)$ communication).
   - This mirrors real-world big data engines (e.g., Apache Spark, Ray, MPI-IO), where data is partitioned across storage/memory nodes and communication is minimized.

### Q8: How does Amdahl's Law explain the speedup curve for small vs. large datasets?
- **Amdahl's Law:** Speedup is capped by the sequential fraction $s$:
  $$S(P) = \frac{1}{s + \frac{1-s}{P}}$$
- For small $N$ ($10^6$ elements), process initialization, barrier synchronization, and IPC communication constitute a significant portion ($s \approx 20\%-50\%$), capping the speedup.
- As problem size scales to $N = 5 \times 10^7$ or $10^8$ elements (**Gustafson's Law** weak scaling), the parallel computational fraction $f_p = (1-s) \to 99.8\%$, enabling near-linear speedup with increasing process counts.

### Q9: Can MPI achieve superlinear speedup ($S(P) > P$)? If so, why?
- Yes, superlinear speedup can occur due to **cache effects**.
- When a single core processes a massive 3 GB vector set, data cannot fit in the CPU's fast L2/L3 caches and must constantly fetch from higher-latency DRAM.
- When partitioned across $P$ ranks, each rank's local buffer ($\approx 3\text{ GB} / P$) may fit entirely within the aggregate L3 cache of all cores, drastically reducing cache misses and memory stalls, leading to an effective speedup greater than $P$.

---

## 4. Code & Practical Implementation Defense

### Q10: How did you ensure 100% numerical correctness?
- Used a deterministic mathematical formula $X[i] = \sin((i \pmod{1000}) \times 0.01) + 1.5$ and $Y[i] = \cos((i \pmod{1000}) \times 0.01) + 2.0$.
- Compared the distributed results on Rank 0 against sequential reference computations with a strict floating-point tolerance of $\epsilon = 10^{-6}$.
- Verified both element-wise array integrity ($Z[i], W[i]$) and global reductions (Dot product, L2 norm, Z sum).

### Q11: Why is `MPI_Wtime()` used for benchmarking instead of standard C `clock()`?
- `clock()` measures processor clock ticks consumed by the calling thread/process, not real wall-clock elapsed time. In multi-threaded or multi-process runs, `clock()` aggregates CPU time across threads.
- `MPI_Wtime()` returns the elapsed wall-clock time in seconds as a high-precision `double` on the calling rank, synchronized with hardware tick counters.

### Q12: Why is `MPI_Barrier(MPI_COMM_WORLD)` placed before `MPI_Wtime()` starts?
- Different MPI processes may launch and reach the starting point at slightly different times due to OS scheduling and process spawn latencies.
- Placing `MPI_Barrier` immediately before the start timer ensures all processes are aligned and synchronized before timing starts, preventing distorted benchmarks.

### Q13: If all 4 threads are fetching from RAM, won't fetching make the RAM busy and create a bottleneck?
- **In light computation (simple addition $X + Y$):** Yes! The math takes only 1 cycle, so all 4 threads finish instantly and simultaneously demand data from RAM, saturating the memory bus. The cores sit stalled waiting for RAM, yielding almost 0× speedup (Memory-Bound).
- **In heavy computation (our code $\sqrt{X^2+Y^2} + \sin X + \cos Y$):** No! The CPU loads a 64-byte block (8 numbers) in a 20-nanosecond burst, and then spends ~300 nanoseconds crunching heavy math. Because computing takes 15× longer than fetching, requests to RAM are staggered and infrequent. The memory bus remains clear, and the CPU hardware prefetcher silently pre-loads the next numbers into local L1 cache in the background (latency hiding). All 4 cores compute simultaneously over 98% of the time, resulting in our 6.59× speedup.
