# Mathematical Formulation, Arithmetic Intensity & Systems Architecture Guide

**Project:** Distributed Vector Processing using Open MPI  
**Course:** Parallel and GPU Computing (PGC Lab)  
**Author:** Akash TD (`01FE24BCI081`)  
**Repository:** [https://github.com/akaraj187/distributed-vector-processing-parallel-computing](https://github.com/akaraj187/distributed-vector-processing-parallel-computing)

---

## 1. Executive Summary: What Problem Are We Solving?

In parallel computing, processing large lists of numbers (vectors) is not just about writing a `for` loop across multiple cores. It is about understanding the balance between **Data Movement (Memory Bus)** and **Mathematical Computation (CPU Arithmetic Units)**.

This guide explains:
1. Every mathematical formula used in our implementation in simple words.
2. Why we intentionally used **heavy arithmetic** ($\sqrt{\dots}$, $\sin$, $\cos$) instead of simple addition ($X + Y$).
3. What real-world scientific, engineering, and AI problems use these exact calculations.

---

## 2. Why Heavy Arithmetic Instead of Simple Addition?

### 2.1 The "Memory Wall" Problem (Why Simple Addition Fails in Parallel Computing)

Suppose we wrote a program that only does simple element-wise addition:
```c
Z[i] = X[i] + Y[i];
```

To compute this single line:
- The CPU must load `X[i]` from RAM: **8 bytes**
- The CPU must load `Y[i]` from RAM: **8 bytes**
- The CPU writes `Z[i]` back to RAM: **8 bytes**
- **Total Memory Transferred:** **24 bytes**
- **Math Performed:** **1 single addition (1 cycle)**

A modern CPU core executes additions at over **3,000,000,000 operations per second** (3 GHz). However, fetching data from main system RAM takes **100 to 200 clock cycles**.

If your code only does simple addition:
- The memory bus gets choked immediately with memory traffic.
- Whether you run on 1 CPU core or 4 CPU cores, the CPU cores spend **95% of their time sitting idle waiting for RAM to deliver numbers**.
- **Result:** You get almost **zero parallel speedup** because the program is completely **Memory-Bound**.

```
Memory-Bound (Simple Addition):
CPU Core 0: [Add] -> [.....WAITING FOR RAM 150 cycles.....] -> [Add]
CPU Core 1: [Add] -> [.....WAITING FOR RAM 150 cycles.....] -> [Add]
                     ^^^^^ Memory bus is jammed! ^^^^^
```

---

### 2.2 Why Heavy Arithmetic Solves This (Compute-Bound Workload)

To properly benchmark parallel CPU cores, we must give each core enough mathematical work so that it actually uses its **Floating-Point Arithmetic Units (FPUs)** instead of waiting for RAM.

We introduced the non-linear mathematical pipeline:
$$W[i] = \sqrt{X[i]^2 + Y[i]^2} + \sin(X[i]) + \cos(Y[i])$$

Notice what happens now:
- The CPU still loads the same 24 bytes of data from memory.
- But now, for each number, the CPU computes:
  1. Two multiplications: $X^2$ and $Y^2$ (2 cycles)
  2. One addition: $X^2 + Y^2$ (1 cycle)
  3. One square root: $\sqrt{\dots}$ (15 to 20 cycles)
  4. One trigonometric sine: $\sin(X)$ (30 to 40 cycles using Taylor polynomial expansion)
  5. One trigonometric cosine: $\cos(Y)$ (30 to 40 cycles)
  6. Final additions (2 cycles)
- **Total Math Operations:** **~90 to 110 CPU clock cycles of arithmetic work per element!**

Now, the CPU cores are actively crunching math rather than waiting for RAM. 
When we distribute this across your laptop's **4 hardware threads**, all 4 cores crunch numbers at full load. This is what transforms the benchmark into a **Compute-Bound workload**, producing real, measurable **4.5× to 6.6× speedups**!

```
Compute-Bound (Heavy Arithmetic):
CPU Core 0: [Squareroot][Sin][Cos][Add][Mult] -> 100% Core Utilization
CPU Core 1: [Squareroot][Sin][Cos][Add][Mult] -> 100% Core Utilization
CPU Core 2: [Squareroot][Sin][Cos][Add][Mult] -> 100% Core Utilization
CPU Core 3: [Squareroot][Sin][Cos][Add][Mult] -> 100% Core Utilization
            ^^^^^ True Parallel Hardware Saturation! ^^^^^
```

---

### 2.3 The Multi-Thread RAM Fetching Mechanism: Why Threads Don't Jam the Memory Bus

A common and critical question in parallel computing is:
> *"If all 4 threads are fetching data from RAM, won't fetching make the RAM busy and create a bottleneck anyway? Do they fetch at the exact same time, and does taking turns create a delay?"*

Here is the exact hardware mechanism that prevents memory bus congestion:

#### 1. When Computation is Very Light (e.g., Simple Addition: $Z = X + Y$)
* The math finishes in **1 single clock cycle** (practically instantaneous).
* All 4 threads finish their computation immediately and **simultaneously demand new data from RAM at the exact same instant**.
* The memory bus and memory controller become saturated and jammed.
* All 4 CPU cores are forced into wait-states (memory stalls), sitting idle while queued for RAM access.
* **Result:** **Memory-Bound Bottleneck** $\to$ Adding more CPU threads provides **almost 0× speedup**.

#### 2. When Computation is Heavy (Our Implementation: $\sqrt{X^2 + Y^2} + \sin X + \cos Y$)
* **Batch Fetching via 64-Byte Cache Lines:** The CPU never fetches 1 number at a time. It loads a 64-byte block containing **8 double-precision numbers in one single 20-nanosecond memory burst**.
* **Staggered, Non-Conflicting Requests:** 
  1. Thread 0 fetches a 64-byte batch in **20 ns**, then immediately begins crunching math on those 8 elements for **~300 ns**.
  2. While Thread 0 is occupied computing, the memory bus is completely free.
  3. In the next 20 ns, Thread 1 fetches its batch $\to$ starts computing for 300 ns.
  4. Thread 2 fetches in 20 ns $\to$ starts computing.
  5. Thread 3 fetches in 20 ns $\to$ starts computing.
* Within just **80 nanoseconds**, all 4 threads have their data and are **all computing simultaneously at 100% core load for the remaining 220+ nanoseconds**!
* **Hardware Prefetching (Latency Hiding):** While each thread is executing heavy math on Batch $K$, the CPU hardware prefetcher silently pre-loads Batch $K+1$ from RAM into the ultra-fast local L1/L2 cache in the background. By the time the thread finishes Batch $K$, the next numbers are already sitting in cache, completely eliminating RAM wait delays.

#### 3. The Buffet Analogy (Why Staggered Fetching Takes < 2% of the Time)
* Imagine 4 people eating dinner:
  * Scooping food from the counter takes **2 seconds**.
  * Sitting down and chewing the meal takes **200 seconds**.
* Person 1 scoops (2s), Person 2 scoops (2s), Person 3 scoops (2s), Person 4 scoops (2s).
* Within 8 seconds, **all 4 people are sitting at their tables eating at the exact same time for the next 192 seconds**.
* Even though they stepped up to the counter one-by-one, **over 96% of the total time is spent with all 4 people eating in parallel**. The counter is never jammed, and no one is starved!

---

## 3. Real-World Applications: What Does This Math Actually Make?

Examiners frequently ask: *"Why did you choose these specific mathematical operations? What are they used for in the real world?"*

Here are the exact industry applications:

### 1. 3D Game Engines & Computer Graphics (Vector Norms & Lighting)
* **The Formula:** $\sqrt{X^2 + Y^2}$
* **Real Application:** In 3D graphics (like Unreal Engine or Blender), every 3D object has millions of surface normal vectors. To calculate how light reflects off a surface, the game engine must normalize every vector (divide by its length $\sqrt{x^2 + y^2 + z^2}$). A modern game normalizes tens of millions of vectors 60 to 120 times every second.

### 2. Machine Learning & AI (Cosine Similarity & Attention)
* **The Formula:** Vector Dot Product $D = \sum (X[i] \cdot Y[i])$
* **Real Application:** When a Large Language Model (like ChatGPT or Gemini) searches for relevant documents using Vector Embeddings, it compares a query vector against millions of document vectors using the **Dot Product** and **Euclidean Norm** (Cosine Similarity). If vector $X$ and vector $Y$ point in the same direction, their dot product is maximized.

### 3. Physics & Wave Simulations (Trigonometric Fields)
* **The Formula:** $\sin(X) + \cos(Y)$
* **Real Application:** Simulating electromagnetic waves, ocean wave modeling, seismic earthquake wave propagation, and acoustics all solve partial differential equations (PDEs) involving wave amplitudes and phases represented by sine and cosine functions.

### 4. High-Performance Computing (BLAS-1 SAXPY)
* **The Formula:** $Z[i] = \alpha \cdot X[i] + \beta \cdot Y[i]$
* **Real Application:** This is the standard **Level-1 BLAS (Basic Linear Algebra Subprograms)** building block. It is used in gradient descent in neural networks ($w_{t+1} = w_t - \eta \cdot \nabla L$), audio signal blending, and linear transformations.

---

## 4. Complete Breakdown of Every Formula in Our Code

### Formula 1: Deterministic Data Generation
$$X[i] = \sin((i \pmod{1000}) \times 0.01) + 1.5$$
$$Y[i] = \cos((i \pmod{1000}) \times 0.01) + 2.0$$
* **Simple Meaning:** Generates predictable floating-point numbers between $[0.5, 2.5]$ for $X$ and $[1.0, 3.0]$ for $Y$.
* **Why:** Avoids random numbers (`rand()`), guaranteeing that the parallel answer matches the single-core answer down to 6 decimal places.

### Formula 2: Linear Combination (SAXPY)
$$Z[i] = 2.5 \cdot X[i] + 1.5 \cdot Y[i]$$
* **Simple Meaning:** Scales list $X$ by 2.5, scales list $Y$ by 1.5, and adds them together into list $Z$.

### Formula 3: Non-Linear Math Pipeline
$$W[i] = \sqrt{X[i]^2 + Y[i]^2} + \sin(X[i]) + \cos(Y[i])$$
* **Simple Meaning:** Calculates Euclidean hypotenuse and wave functions for each element to saturate the CPU arithmetic units.

### Formula 4: Global Dot Product
$$D = \sum_{i=0}^{N-1} X[i] \cdot Y[i]$$
* **Simple Meaning:** Multiplies matching pairs and sums them into a single scalar number.

### Formula 5: Euclidean L2 Norm
$$\|X\|_2 = \sqrt{\sum_{i=0}^{N-1} X[i]^2}$$
* **Simple Meaning:** Calculates the total geometric magnitude of the vector array.

### Formula 6: Domain Decomposition with Remainder Handling
$$n_{\text{local}}(r) = \begin{cases} \lfloor N/P \rfloor + 1, & \text{if } r < (N \pmod P) \\ \lfloor N/P \rfloor, & \text{otherwise} \end{cases}$$
* **Simple Meaning:** Splits vector $N$ evenly across $P$ processes. Any leftover numbers ($N \pmod P$) are given one-by-one to the first few processes so no numbers are dropped.

### Formula 7: Speedup and Parallel Efficiency
$$\text{Speedup } S(P) = \frac{T_{\text{Sequential}}}{T_{\text{MPI}}(P)}$$
$$\text{Efficiency } E(P) = \frac{S(P)}{P} \times 100\%$$
* **Speedup:** Tells you how many times faster your parallel MPI program ran compared to 1 core (e.g., **6.59× faster** on our machine).
* **Efficiency:** Tells you what percentage of your CPU core investment is directly translated into speed.
