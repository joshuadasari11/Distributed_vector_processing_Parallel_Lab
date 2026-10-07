#!/usr/bin/env python3
"""
Performance Visualization and Scaling Analysis
Distributed Vector Processing - MPI Benchmark Analysis
Parallel & GPU Computing Lab - Team 5
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Apply clean plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 16

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_FILE = os.path.join(ROOT_DIR, "results", "timing_results.csv")
GRAPHS_DIR = os.path.join(ROOT_DIR, "graphs")

os.makedirs(GRAPHS_DIR, exist_ok=True)

def load_data():
    records = []
    with open(CSV_FILE, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'Paradigm': row['Paradigm'],
                'Mode': row['Mode'],
                'N': int(row['N']),
                'Processes': int(row['Processes']),
                'TotalTime': float(row['TotalTime']),
                'CommTime': float(row['CommTime']),
                'CompTime': float(row['CompTime']),
                'ScatterTime': float(row['ScatterTime']),
                'ReduceTime': float(row['ReduceTime']),
                'GatherTime': float(row['GatherTime']),
                'Throughput': float(row['Throughput_M_elems_per_sec']),
                'Speedup': float(row['Speedup']),
                'Efficiency': float(row['Efficiency']),
                'Status': row['Status']
            })
    return records

def generate_plots():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please run benchmarks first.")
        return

    data = load_data()
    print(f"Loaded {len(data)} benchmark records from {CSV_FILE}")

    sizes = sorted(list(set(r['N'] for r in data)))
    processes = sorted(list(set(r['Processes'] for r in data if r['Processes'] > 0)))

    # -------------------------------------------------------------------------
    # Plot 1: Execution Time vs Number of MPI Processes (In-Situ vs ScatterGather)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # In-Situ Scaling
    ax1 = axes[0]
    for n in sizes:
        sub = sorted([r for r in data if r['N'] == n and r['Mode'] == 'InSitu'], key=lambda x: x['Processes'])
        p_vals = [r['Processes'] for r in sub]
        t_vals = [r['TotalTime'] for r in sub]
        label = f"N = {n/1e6:.0f}M elements"
        ax1.plot(p_vals, t_vals, marker='o', linewidth=2.2, label=label)

    ax1.set_title("In-Situ Distributed Partitioning\n(Domain Decomposed)", fontweight='bold')
    ax1.set_xlabel("Number of MPI Processes (P)")
    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_xticks(processes)
    ax1.set_yscale('log')
    ax1.legend(frameon=True, facecolor='#f8f9fa')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Scatter-Gather Scaling
    ax2 = axes[1]
    for n in sizes:
        sub = sorted([r for r in data if r['N'] == n and r['Mode'] == 'ScatterGather'], key=lambda x: x['Processes'])
        p_vals = [r['Processes'] for r in sub]
        t_vals = [r['TotalTime'] for r in sub]
        label = f"N = {n/1e6:.0f}M elements"
        ax2.plot(p_vals, t_vals, marker='s', linewidth=2.2, label=label)

    ax2.set_title("Centralized Scatter-Gather Workflow\n(Master-Worker IPC Communication)", fontweight='bold')
    ax2.set_xlabel("Number of MPI Processes (P)")
    ax2.set_ylabel("Execution Time (seconds)")
    ax2.set_xticks(processes)
    ax2.set_yscale('log')
    ax2.legend(frameon=True, facecolor='#f8f9fa')
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.suptitle("Distributed Vector Processing: Execution Time vs. Process Count", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot1_path = os.path.join(GRAPHS_DIR, "execution_time_vs_processes.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot1_path}")

    # -------------------------------------------------------------------------
    # Plot 2: Speedup Analysis (Strong Scaling)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # In-Situ Speedup
    ax1 = axes[0]
    ax1.plot(processes, processes, 'k--', linewidth=1.8, label="Ideal Linear Speedup (S=P)")
    for n in sizes:
        sub = sorted([r for r in data if r['N'] == n and r['Mode'] == 'InSitu'], key=lambda x: x['Processes'])
        p_vals = [r['Processes'] for r in sub]
        s_vals = [r['Speedup'] for r in sub]
        ax1.plot(p_vals, s_vals, marker='o', linewidth=2.2, label=f"N = {n/1e6:.0f}M elements")

    ax1.set_title("In-Situ Speedup $S(P) = T_{seq} / T_P$", fontweight='bold')
    ax1.set_xlabel("MPI Processes (P)")
    ax1.set_ylabel("Speedup Factor")
    ax1.set_xticks(processes)
    ax1.legend(frameon=True, facecolor='#f8f9fa')
    ax1.grid(True, ls="--", alpha=0.5)

    # Scatter-Gather Speedup
    ax2 = axes[1]
    ax2.plot(processes, processes, 'k--', linewidth=1.8, label="Ideal Linear Speedup (S=P)")
    for n in sizes:
        sub = sorted([r for r in data if r['N'] == n and r['Mode'] == 'ScatterGather'], key=lambda x: x['Processes'])
        p_vals = [r['Processes'] for r in sub]
        s_vals = [r['Speedup'] for r in sub]
        ax2.plot(p_vals, s_vals, marker='s', linewidth=2.2, label=f"N = {n/1e6:.0f}M elements")

    ax2.set_title("Centralized Scatter-Gather Speedup", fontweight='bold')
    ax2.set_xlabel("MPI Processes (P)")
    ax2.set_ylabel("Speedup Factor")
    ax2.set_xticks(processes)
    ax2.legend(frameon=True, facecolor='#f8f9fa')
    ax2.grid(True, ls="--", alpha=0.5)

    plt.suptitle("Speedup Analysis across Workload Scales", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot2_path = os.path.join(GRAPHS_DIR, "speedup_analysis.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot2_path}")

    # -------------------------------------------------------------------------
    # Plot 3: Parallel Efficiency Analysis
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.axhline(100, color='black', linestyle='--', linewidth=1.5, label="100% Peak Ideal Efficiency")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, n in enumerate(sizes):
        sub_insitu = sorted([r for r in data if r['N'] == n and r['Mode'] == 'InSitu'], key=lambda x: x['Processes'])
        p_vals = [r['Processes'] for r in sub_insitu]
        e_vals = [r['Efficiency'] for r in sub_insitu]
        plt.plot(p_vals, e_vals, marker='o', linewidth=2.2,
                 color=colors[idx % len(colors)], label=f"In-Situ (N={n/1e6:.0f}M)")

    plt.title("Parallel Efficiency vs. MPI Processes ($E(P) = S(P) / P \\times 100\\%$)", fontweight='bold')
    plt.xlabel("Number of MPI Processes (P)")
    plt.ylabel("Parallel Efficiency (%)")
    plt.xticks(processes)
    plt.legend(frameon=True, facecolor='#f8f9fa')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plot3_path = os.path.join(GRAPHS_DIR, "parallel_efficiency.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot3_path}")

    # -------------------------------------------------------------------------
    # Plot 4: Data Size Scaling (Log-Log Scale)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Sequential
    seq_records = sorted([r for r in data if r['Paradigm'] == 'Sequential'], key=lambda x: x['N'])
    plt.plot([r['N'] / 1e6 for r in seq_records], [r['TotalTime'] for r in seq_records],
             marker='o', color='#333333', linewidth=2.5, label="Sequential Baseline (1 core)")

    # MPI In-Situ P=2, 4, 8, 16
    mpi_p_selected = [2, 4, 8, 16]
    for p in mpi_p_selected:
        sub = sorted([r for r in data if r['Mode'] == 'InSitu' and r['Processes'] == p], key=lambda x: x['N'])
        if len(sub) > 0:
            plt.plot([r['N'] / 1e6 for r in sub], [r['TotalTime'] for r in sub],
                     marker='^', linewidth=2.0, label=f"MPI In-Situ ({p} processes)")

    plt.title("Execution Time Scaling with Vector Size $N$ ($10^6$ to $5\\times 10^7$ Elements)", fontweight='bold')
    plt.xlabel("Vector Size $N$ (Millions of elements)")
    plt.ylabel("Total Execution Time (seconds)")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(frameon=True, facecolor='#f8f9fa')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plot4_path = os.path.join(GRAPHS_DIR, "datasize_scaling.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot4_path}")

    # -------------------------------------------------------------------------
    # Plot 5: Computation vs Communication Breakdown (Centralized Model)
    # -------------------------------------------------------------------------
    target_n = sizes[-1]  # 50M
    sub_sg = sorted([r for r in data if r['N'] == target_n and r['Mode'] == 'ScatterGather'], key=lambda x: x['Processes'])

    if len(sub_sg) > 0:
        plt.figure(figsize=(10, 6))
        bar_width = 0.55
        x_indices = np.arange(len(sub_sg))

        comp_times = np.array([r['CompTime'] for r in sub_sg])
        scatter_times = np.array([r['ScatterTime'] for r in sub_sg])
        gather_times = np.array([r['GatherTime'] for r in sub_sg])
        reduce_times = np.array([r['ReduceTime'] for r in sub_sg])

        plt.bar(x_indices, comp_times, bar_width, label='Pure Computation', color='#2ca02c')
        plt.bar(x_indices, scatter_times, bar_width, bottom=comp_times, label='MPI_Scatterv (Data Distribution)', color='#1f77b4')
        plt.bar(x_indices, gather_times, bar_width, bottom=comp_times + scatter_times, label='MPI_Gatherv (Result Collection)', color='#ff7f0e')
        plt.bar(x_indices, reduce_times, bar_width, bottom=comp_times + scatter_times + gather_times, label='MPI_Reduce (Global Reductions)', color='#d62728')

        plt.title(f"Communication vs. Computation Breakdown (N = {target_n/1e6:.0f}M elements)", fontweight='bold')
        plt.xlabel("MPI Processes (P)")
        plt.ylabel("Time (seconds)")
        plt.xticks(x_indices, [f"P={r['Processes']}" for r in sub_sg])
        plt.legend(frameon=True, facecolor='#f8f9fa')
        plt.grid(True, ls="--", alpha=0.4, axis='y')
        plt.tight_layout()
        plot5_path = os.path.join(GRAPHS_DIR, "computation_vs_communication.png")
        plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot5_path}")

    print("All 5 performance visualization graphs generated successfully!")

if __name__ == "__main__":
    generate_plots()
