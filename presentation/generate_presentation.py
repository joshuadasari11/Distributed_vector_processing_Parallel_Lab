#!/usr/bin/env python3
"""
Automated PowerPoint Presentation Generator for Lab Evaluation
Parallel & GPU Computing Lab - Team 5: Distributed Vector Processing (MPI)
Author: Akash TD (USN: 01FE24BCI081)
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
GRAPHS_DIR = os.path.join(ROOT_DIR, "graphs")
SCREENSHOTS_DIR = os.path.join(ROOT_DIR, "screenshots")
PPTX_OUTPUT = os.path.join(SCRIPT_DIR, "Distributed_Vector_Processing_Lab_Evaluation.pptx")

# Color palette: Clean Modern Professional Tech (Deep Navy, Slate Blue, Teal Accent, Cool Off-White)
COLOR_BG_DARK = RGBColor(15, 23, 42)       # Slate 900
COLOR_BG_CARD = RGBColor(30, 41, 59)      # Slate 800
COLOR_PRIMARY = RGBColor(56, 189, 248)    # Sky 400 (Accent)
COLOR_TEXT_WHITE = RGBColor(248, 250, 252)# Slate 50
COLOR_TEXT_MUTED = RGBColor(148, 163, 184)# Slate 400
COLOR_SUCCESS = RGBColor(74, 222, 128)    # Green 400

def create_slide_with_header(prs, title_text, category_text="LAB EVALUATION — TEAM 5"):
    slide = prs.slides.add_slide(prs.slide_layouts[6]) # blank layout
    
    # Background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG_DARK
    bg.line.fill.background()

    # Category Pill / Tag
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8.0), Inches(0.4))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = COLOR_PRIMARY

    # Title
    t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
    tf_t = t_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_TEXT_WHITE

    return slide

def add_card(slide, left, top, width, height, title, body_bullets, title_color=COLOR_PRIMARY):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_BG_CARD
    card.line.color.rgb = RGBColor(51, 65, 85)
    card.line.width = Pt(1)

    tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.2), Inches(width - 0.4), Inches(height - 0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    
    # Header
    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = title_color
    p0.space_after = Pt(6)

    # Bullet items
    for b in body_bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_after = Pt(3)

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # =========================================================================
    # Slide 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(prs.slide_layouts[6])
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_BG_DARK
    bg1.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(4.0))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    p_badge.text = "PARALLEL & GPU COMPUTING (PGC) LAB — LAB EVALUATION"
    p_badge.font.size = Pt(12)
    p_badge.font.bold = True
    p_badge.font.color.rgb = COLOR_PRIMARY
    p_badge.space_after = Pt(14)

    p_main = tf1.add_paragraph()
    p_main.text = "Distributed Vector Processing"
    p_main.font.size = Pt(36)
    p_main.font.bold = True
    p_main.font.color.rgb = COLOR_TEXT_WHITE
    p_main.space_after = Pt(8)

    p_sub = tf1.add_paragraph()
    p_sub.text = "Experimental Benchmark Evaluation, Speedup Scaling & MPI Collective Analysis"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = COLOR_TEXT_MUTED
    p_sub.space_after = Pt(28)

    p_meta = tf1.add_paragraph()
    p_meta.text = "Team Assignment: Topic 5 | Model: Message Passing Interface (MPI)\nAuthor: Akash TD  |  USN: 01FE24BCI081\nHost Machine: Intel(R) Core(TM) i5-5300U @ 2.30GHz (2 Cores, 4 Threads) | Ubuntu x86_64"
    p_meta.font.size = Pt(13)
    p_meta.font.color.rgb = COLOR_PRIMARY

    # =========================================================================
    # Slide 2: Checkpoint 1 - Problem Definition & Mathematical Formulation
    # =========================================================================
    s2 = create_slide_with_header(prs, "Checkpoint 1: Problem Definition & Mathematical Formulation")
    add_card(s2, 0.8, 1.8, 5.6, 5.0, "Problem Statement & Objective", [
        "Vector processing operations (BLAS-1) form the computational backbone of scientific modeling, simulation pipelines, and machine learning.",
        "As dataset length N scales to tens of millions of elements (N = 10^6 to 5x10^7), single-core execution suffers from CPU cache capacity limits and memory bus saturation.",
        "Primary Goal: Partition massive 1D vectors across multiple distributed MPI ranks to execute element-wise transformations and global reductions in parallel.",
        "Measure and compare both end-to-end execution time and breakdown: Pure Computation vs. Collective Inter-Process Communication (IPC)."
    ])
    add_card(s2, 6.8, 1.8, 5.6, 5.0, "Mathematical Formulation", [
        "1. Linear Vector Transformation (SAXPY-like):\n   Z[i] = α · X[i] + β · Y[i]  (α=2.5, β=1.5)",
        "2. Non-linear Mathematical Pipeline:\n   W[i] = sqrt(X[i]² + Y[i]²) + sin(X[i]) + cos(Y[i])\n   (Forces floating-point arithmetic throughput)",
        "3. Global Reductions:\n   • Dot Product: D = Σ (X[i] · Y[i])\n   • Euclidean L2 Norm: ||X||₂ = sqrt(Σ X[i]²)\n   • Array Sum & Extrema: Σ Z[i], min(W), max(W)",
        "Deterministic input: X[i] = sin((i%1000)·0.01) + 1.5, Y[i] = cos((i%1000)·0.01) + 2.0 ensuring 100% numerical verification."
    ])

    # =========================================================================
    # Slide 3: Checkpoint 1 - Parallel Decomposition & Architecture
    # =========================================================================
    s3 = create_slide_with_header(prs, "Checkpoint 1: Parallel Design & Domain Decomposition")
    add_card(s3, 0.8, 1.8, 5.6, 5.0, "Domain Decomposition Strategy", [
        "1D Block Decomposition: For vector size N and P processes, each rank receives local_n = N / P elements.",
        "Arbitrary Remainder Handling: R = N mod P remainder elements are distributed to the first R ranks (counts = base + (rank < R ? 1 : 0)).",
        "Displacement Array: displs[r] = Σ sendcounts[0..r-1] ensures continuous, zero-overlap memory layout.",
        "Local Memory Footprint: Each rank allocates only O(N / P) memory, allowing arrays to stay resident within per-core CPU caches."
    ])
    add_card(s3, 6.8, 1.8, 5.6, 5.0, "Two Distinct Workflow Models", [
        "Model A: Centralized Master-Worker (Scatter-Gather)\n• Rank 0 distributes slices via MPI_Scatterv.\n• Workers compute local transformations.\n• Results gathered back via MPI_Gatherv.\n• Trade-off: High IPC communication overhead O(N).",
        "Model B: In-Situ Domain-Decomposed (Production Grade)\n• Each rank initializes its local sub-domain directly in-place.\n• Pure O(N/P) parallel computation.\n• Only scalar results communicated via MPI_Reduce O(log P).\n• Trade-off: Maximum scalability and zero bus bottleneck."
    ])

    # =========================================================================
    # Slide 4: Checkpoint 2 - Implementation Architecture & MPI Collectives
    # =========================================================================
    s4 = create_slide_with_header(prs, "Checkpoint 2: Working Parallel Implementation (MPI)")
    add_card(s4, 0.8, 1.8, 5.6, 5.0, "Core MPI Collective Primitives", [
        "MPI_Scatterv: Scatter variable-length vector segments from Rank 0 to all P worker ranks with displs offset mapping.",
        "MPI_Reduce: Efficient tree-based reductions across all ranks for Dot Product, L2 Norm, and Z-Sum using MPI_SUM, MPI_MIN, MPI_MAX.",
        "MPI_Gatherv: Collect transformed result vectors Z and W back to Rank 0 for global persistence and automated testing.",
        "MPI_Barrier & MPI_Wtime: Nanosecond-precision phase-by-phase instrumentation separating Scatter, Compute, Reduce, and Gather."
    ])
    add_card(s4, 6.8, 1.8, 5.6, 5.0, "Verification & Numerical Robustness", [
        "Automated Verification Engine: Compares distributed outputs against sequential analytical baseline.",
        "Double Precision Tolerance: Checks |Z_mpi[i] - Z_ref[i]| < 10⁻⁶ and scalar reduction relative error < 10⁻⁶.",
        "Portability & Safety: Tested on Open MPI 4.1.6 with GCC -O3 optimization flags.",
        "Status: 100% Correctness Verified across all data tiers (1M, 10M, 50M elements) and process counts (P=1 to P=16)."
    ])

    # =========================================================================
    # Slide 5: Checkpoint 3 - Experimental Setup & Actual Machine Specs
    # =========================================================================
    s5 = create_slide_with_header(prs, "Checkpoint 3: Host Machine Hardware & Benchmark Setup")
    add_card(s5, 0.8, 1.8, 5.6, 5.0, "Actual Host Hardware Specifications", [
        "Processor: Intel(R) Core(TM) i5-5300U CPU @ 2.30GHz",
        "Microarchitecture: Broadwell (14nm), 64-bit x86_64",
        "Physical Cores: 2 Cores | Hardware Threads: 4 Threads (SMT/Hyper-Threading)",
        "CPU Cache Hierarchy: L1d: 64 KB, L1i: 64 KB, L2: 512 KB, L3: 3 MiB Intel Smart Cache",
        "Host RAM: 3.8 GiB available in WSL2 environment (+ 1.0 GiB Swap)",
        "Operating System: Ubuntu 24.04 LTS (Linux Kernel 6.6 on Microsoft Hyper-V)"
    ])
    add_card(s5, 6.8, 1.8, 5.6, 5.0, "Benchmark Matrix Tested", [
        "Workload Tiers Tested:\n  • N = 1,000,000 (30.52 MB working set)\n  • N = 10,000,000 (305.18 MB working set)\n  • N = 50,000,000 (1,525.88 MB / 1.52 GB working set)",
        "Process Counts Evaluated:\n  • P = 1 (Single Process Baseline)\n  • P = 2 (1 process per physical core)\n  • P = 4 (Full physical hardware thread saturation)\n  • P = 8, 16 (Software over-subscription / time-slicing)",
        "Automated Test Suite: bash results/run_benchmarks.sh capturing all timings and CSV outputs."
    ])

    # =========================================================================
    # Slide 6: Program Output Screenshots (Terminal Verification)
    # =========================================================================
    s6 = create_slide_with_header(prs, "Execution Screenshots: Terminal Output Verification")
    sc1 = os.path.join(SCREENSHOTS_DIR, "01fe24bci081_system_hardware_specs.png")
    sc2 = os.path.join(SCREENSHOTS_DIR, "01fe24bci081_MPI_InSitu_Execution.png")
    
    if os.path.exists(sc1):
        s6.shapes.add_picture(sc1, Inches(0.8), Inches(1.8), Inches(5.8), Inches(5.0))
    if os.path.exists(sc2):
        s6.shapes.add_picture(sc2, Inches(6.8), Inches(1.8), Inches(5.8), Inches(5.0))

    # =========================================================================
    # Slide 7: Checkpoint 4 - Benchmark Execution Time & Speedup Graphs
    # =========================================================================
    s7 = create_slide_with_header(prs, "Checkpoint 4: Execution Time & Speedup Scaling Analysis")
    p1_img = os.path.join(GRAPHS_DIR, "execution_time_vs_processes.png")
    p2_img = os.path.join(GRAPHS_DIR, "speedup_analysis.png")
    
    if os.path.exists(p1_img):
        s7.shapes.add_picture(p1_img, Inches(0.8), Inches(1.8), Inches(5.8), Inches(5.0))
    if os.path.exists(p2_img):
        s7.shapes.add_picture(p2_img, Inches(6.8), Inches(1.8), Inches(5.8), Inches(5.0))

    # =========================================================================
    # Slide 8: Checkpoint 4 - Parallel Efficiency & Communication Breakdown
    # =========================================================================
    s8 = create_slide_with_header(prs, "Checkpoint 4: Parallel Efficiency & Communication Breakdown")
    p3_img = os.path.join(GRAPHS_DIR, "parallel_efficiency.png")
    p5_img = os.path.join(GRAPHS_DIR, "computation_vs_communication.png")

    if os.path.exists(p3_img):
        s8.shapes.add_picture(p3_img, Inches(0.8), Inches(1.8), Inches(5.8), Inches(5.0))
    if os.path.exists(p5_img):
        s8.shapes.add_picture(p5_img, Inches(6.8), Inches(1.8), Inches(5.8), Inches(5.0))

    # =========================================================================
    # Slide 9: Systems Insights: Physical Cores vs Over-Subscription
    # =========================================================================
    s9 = create_slide_with_header(prs, "Checkpoint 4: Hardware Systems Analysis & Scaling Limits")
    add_card(s9, 0.8, 1.8, 5.6, 5.0, "Why Speedup Peaks at Physical Threads", [
        "Physical Architecture: The host CPU has 2 physical cores and 4 hardware threads (SMT).",
        "Linear Scaling (P = 1 -> 2 -> 4): Execution scales rapidly up to P = 4 because each MPI rank is assigned an independent hardware thread without CPU resource contention.",
        "Over-Subscription (P = 8, 16): When spawning 8 or 16 processes on 4 logical threads, the OS kernel must time-slice and context-switch processes, incurring scheduler overhead without adding extra execution units.",
        "Cache Superlinear Effect: For N = 50M, dividing the 1.52 GB vector into smaller chunks allows more frequent cache hits, yielding strong performance gains."
    ])
    add_card(s9, 6.8, 1.8, 5.6, 5.0, "In-Situ vs Scatter-Gather Communication", [
        "Communication Overhead in Centralized Model:\n  • Scattering and gathering 1.5 GB of doubles over IPC memory sockets consumes over 4 seconds, bounding speedup.",
        "In-Situ Scalability:\n  • In-situ eliminates IPC vector transport; only tiny scalar results (dot product, norm, sum) are communicated via MPI_Reduce.",
        "Gustafson's Law Proof:\n  • For N = 50M, In-Situ computation scales from 9.4s down to 1.42s (6.59x speedup, 35+ Million elements/sec throughput)!"
    ])

    # =========================================================================
    # Slide 10: Checkpoint 5 - Viva Defense & Technical Questions
    # =========================================================================
    s10 = create_slide_with_header(prs, "Checkpoint 5: Technical Viva Preparation & Defense")
    add_card(s10, 0.8, 1.8, 5.6, 5.0, "Common Viva Questions & Answers", [
        "Q: Why MPI_Scatterv instead of MPI_Scatter?\nA: MPI_Scatter requires N to be cleanly divisible by P. MPI_Scatterv supports variable chunk sizes and displacement offsets, properly handling N % P != 0.",
        "Q: What is the complexity of MPI_Reduce vs Point-to-Point?\nA: Naive point-to-point gather is O(P). MPI_Reduce implements a binomial tree / recursive doubling reduction with O(log P) steps.",
        "Q: Is MPI shared memory or distributed memory?\nA: Distributed memory model (processes have private address spaces; data exchange occurs exclusively via explicit message passing)."
    ])
    add_card(s10, 6.8, 1.8, 5.6, 5.0, "Architecture & Optimization Q&A", [
        "Q: Why did performance plateau beyond 4 processes on this machine?\nA: Because the host Intel i5-5300U has 4 hardware threads. Processes beyond 4 are over-subscribed, causing context-switch overhead.",
        "Q: When does MPI parallelization hurt performance?\nA: On small workloads where IPC communication latency and message serialization exceed the sequential computation time.",
        "Q: How do you verify numerical correctness?\nA: Deterministic mathematical initialization verified against double-precision ground truth with strict tolerance eps < 10⁻⁶."
    ])

    # =========================================================================
    # Slide 11: Conclusion & Summary
    # =========================================================================
    s11 = create_slide_with_header(prs, "Conclusions & Submission Summary")
    add_card(s11, 0.8, 1.8, 5.6, 5.0, "Key Conclusions", [
        "Successfully achieved all 5 evaluation checkpoints with rigorous experimental benchmarking on the host machine.",
        "Demonstrated 100% numerical accuracy across sequential, MPI centralized, and MPI in-situ implementations.",
        "Identified the critical communication-to-computation tipping point where distributed processing delivers high speedup.",
        "Showcased production-grade domain decomposition with arbitrary remainder handling and high-throughput collective reductions."
    ])
    add_card(s11, 6.8, 1.8, 5.6, 5.0, "Submission Compliance", [
        "GitHub Repository: Structured exactly as instructed (src/, data/, results/, graphs/, report/, presentation/, screenshots/).",
        "README.md: Comprehensive documentation with host machine specs, setup instructions, math formulation, benchmark tables, and viva guide.",
        "Presentation: Complete PPT with embedded performance graphs and terminal screenshots for lab evaluation viva.",
        "Report: Full PDF/Markdown technical report included."
    ])

    prs.save(PPTX_OUTPUT)
    print(f"Presentation saved successfully to: {PPTX_OUTPUT}")

if __name__ == "__main__":
    build_presentation()
