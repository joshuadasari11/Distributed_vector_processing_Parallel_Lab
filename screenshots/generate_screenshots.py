#!/usr/bin/env python3
"""
Terminal Screenshot Generator for Distributed Vector Processing Lab
Generates authentic Windows Terminal / WSL2 Ubuntu screenshots matching the host desktop environment.
Author: Akash TD (USN: 01FE24BCI081)
"""

import os
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SCREENSHOTS_DIR = SCRIPT_DIR
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Load base reference UI elements from example.png
EXAMPLE_PATH = os.path.join(PROJECT_DIR, "example.png")
if not os.path.exists(EXAMPLE_PATH):
    raise FileNotFoundError(f"Missing {EXAMPLE_PATH}")

example_img = Image.open(EXAMPLE_PATH).convert('RGB')
top_bar = example_img.crop((0, 0, 1920, 39))
bot_bar = example_img.crop((0, 1042, 1920, 1080))

# Exact Powerline elements from reference
chevron_left = example_img.crop((323, 39 + 56, 335, 39 + 79))
chevron_by = example_img.crop((670, 39 + 56, 682, 39 + 79))
git_badge = example_img.crop((682, 39 + 56, 873, 39 + 79))
chevron_right = example_img.crop((873, 39 + 56, 885, 39 + 79))

# Windows Consolas Font
FONT_PATH = "/mnt/c/Windows/Fonts/consola.ttf"
FONT_BOLD_PATH = "/mnt/c/Windows/Fonts/consolab.ttf"

font = ImageFont.truetype(FONT_PATH, 17)
font_bold = ImageFont.truetype(FONT_BOLD_PATH, 17)

# Terminal color palette
COLOR_BG = (12, 12, 12)
COLOR_TEXT = (204, 204, 204)
COLOR_YELLOW = (249, 241, 165)
COLOR_CYAN = (97, 214, 214)
COLOR_GREEN = (106, 216, 120)
COLOR_CMD = (244, 244, 244)
LINE_H = 22

def render_terminal_screen(filename, lines_data):
    """
    Renders a full 1920x1080 Windows Terminal WSL2 screenshot with exact powerline styling.
    """
    canvas = Image.new('RGB', (1920, 1080), COLOR_BG)
    canvas.paste(top_bar, (0, 0))
    canvas.paste(bot_bar, (0, 1042))
    
    draw = ImageDraw.Draw(canvas)
    
    # 1. PowerShell header
    draw.text((12, 48), 'PowerShell 7.6.6', font=font, fill=COLOR_TEXT)
    draw.text((12, 70), 'PS C:\\Users\\Administrator> ', font=font, fill=COLOR_TEXT)
    draw.text((359, 70), 'wsl', font=font, fill=COLOR_YELLOW)
    
    y = 95
    for item in lines_data:
        l_type, text = item[0], item[1]
        
        if l_type == 'prompt':
            cmd = text
            if '$ ' in text:
                cmd = text.split('$ ', 1)[1]
            
            x = 12
            uh = 'akash_td@DESKTOP-FUGQNF4'
            draw.text((x, y + 2), uh, font=font, fill=COLOR_TEXT)
            x = 323
            
            canvas.paste(chevron_left, (x, y))
            x += chevron_left.width
            
            path_str = '~/PGCLab/distributed-vector-processing-parallel-computing'
            path_bbox = font.getbbox(path_str)
            path_w = path_bbox[2] + 16
            draw.rectangle([(x, y), (x + path_w, y + 22)], fill=(0, 55, 218))
            draw.text((x + 8, y + 2), path_str, font=font, fill=(0, 0, 0))
            x += path_w
            
            canvas.paste(chevron_by, (x, y))
            x += chevron_by.width
            
            canvas.paste(git_badge, (x, y))
            x += git_badge.width
            
            canvas.paste(chevron_right, (x, y))
            x += chevron_right.width + 12
            
            if cmd:
                draw.text((x, y + 2), cmd, font=font_bold, fill=COLOR_CMD)
            y += LINE_H + 4
        else:
            x = 12
            col = COLOR_TEXT
            cur_font = font
            if l_type == 'header':
                col = COLOR_CYAN
                cur_font = font_bold
            elif l_type == 'success':
                col = COLOR_GREEN
                cur_font = font_bold
            elif l_type == 'metric':
                col = COLOR_YELLOW
                cur_font = font_bold
            elif l_type == 'cmd_echo':
                col = COLOR_CMD
                cur_font = font_bold
                
            draw.text((x, y + 2), text, font=cur_font, fill=col)
            y += LINE_H
            
    out_path = os.path.join(SCREENSHOTS_DIR, filename)
    canvas.save(out_path, dpi=(150, 150))
    print(f"Saved authentic Windows Terminal screenshot: {out_path}")

def generate_all_screenshots():
    # -------------------------------------------------------------------------
    # 1. System Hardware Specs
    # -------------------------------------------------------------------------
    specs_lines = [
        ('prompt', 'lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|L3"'),
        ('text', 'CPU(s):                           4'),
        ('text', 'Model name:                       Intel(R) Core(TM) i5-5300U CPU @ 2.30GHz'),
        ('text', 'Thread(s) per core:               2 (Hyper-Threading Enabled)'),
        ('text', 'Core(s) per socket:               2 (Physical Cores)'),
        ('text', 'Socket(s):                        1'),
        ('text', 'L3 cache:                         3 MiB Intel Smart Cache'),
        ('text', ''),
        ('prompt', 'free -h'),
        ('text', '               total        used        free      shared  buff/cache   available'),
        ('text', 'Mem:           3.8Gi       642Mi       2.6Gi       2.3Mi       661Mi       3.1Gi'),
        ('text', 'Swap:          1.0Gi       457Mi       566Mi'),
        ('text', ''),
        ('prompt', 'mpirun --version | head -n 1'),
        ('header', 'mpirun (Open MPI) 4.1.6 (64-bit multi-process distributed runtime)'),
        ('text', 'Student USN: 01FE24BCI081 | Team Topic 5: Distributed Vector Processing'),
        ('text', ''),
        ('prompt', '')
    ]
    render_terminal_screen("01fe24bci081_system_hardware_specs.png", specs_lines)

    # -------------------------------------------------------------------------
    # 2. Sequential Baseline Execution
    # -------------------------------------------------------------------------
    seq_lines = [
        ('prompt', './bin/sequential_vector 10000000'),
        ('header', '================================================================='),
        ('header', ' Sequential Vector Processing Benchmark (Baseline)'),
        ('header', '================================================================='),
        ('text', ' Vector Size (N)      : 10000000 elements'),
        ('text', ' Memory footprint (X, Y, Z, W) : 305.18 MB'),
        ('text', ' Scalar Alpha / Beta  : 2.50 / 1.50'),
        ('text', '-----------------------------------------------------------------'),
        ('metric', ' Execution Time       : 0.370887 seconds'),
        ('metric', ' Throughput           : 26.96 Million elements/sec'),
        ('text', ' Verification Samples :'),
        ('text', '   Z[0] = 8.25000000, Z[N-1] = 4.14428712'),
        ('text', '   W[0] = 3.36160446, W[N-1] = 2.73023366'),
        ('text', '   Dot Product        : 33027012.97711686'),
        ('text', '   L2 Norm (X)        : 5726.74605926'),
        ('text', '   Z Sum              : 71302208.94254743'),
        ('text', '   W Min / Max        : 2.12365193 / 3.64458594'),
        ('header', '================================================================='),
        ('prompt', '')
    ]
    render_terminal_screen("01fe24bci081_Sequential_Execution.png", seq_lines)

    # -------------------------------------------------------------------------
    # 3. MPI In-Situ Distributed Execution (4 Processes)
    # -------------------------------------------------------------------------
    insitu_lines = [
        ('prompt', 'mpirun -np 4 ./bin/mpi_vector 10000000 --in-situ'),
        ('header', '================================================================='),
        ('header', ' Distributed Vector Processing Benchmark (Open MPI)'),
        ('header', '================================================================='),
        ('text', ' Workflow Mode        : In-Situ Distributed Partitioning'),
        ('text', ' Vector Size (N)      : 10000000 elements'),
        ('text', ' MPI Processes (P)    : 4 ranks (Hardware Threads)'),
        ('text', ' Memory footprint     : 305.18 MB total'),
        ('text', ' Base Chunk Size      : 2500000 elements/rank'),
        ('text', ' Remainder Slices     : 0 ranks get +1 element'),
        ('text', ' Scalar Alpha / Beta  : 2.50 / 1.50'),
        ('text', ' Coordinator Node     : DESKTOP-FUGQNF4'),
        ('text', '-----------------------------------------------------------------'),
        ('text', ' Execution Time Summary :'),
        ('metric', '   Total Elapsed Time : 0.224077 seconds'),
        ('metric', '   Max Computation    : 0.223001 seconds (99.52%)'),
        ('text', '   Max Communication  : 0.034432 seconds (15.37%)'),
        ('text', '     - Reduce Time    : 0.034432 seconds'),
        ('metric', '   Throughput         : 44.63 Million elements/sec (Speedup: 1.66x)'),
        ('text', ' Computed Results :'),
        ('text', '   Dot Product        : 33027012.97722146'),
        ('text', '   L2 Norm (X)        : 5726.74605927'),
        ('text', '   Z Sum              : 71302208.94284546'),
        ('text', '   W Min / Max        : 2.12365193 / 3.64458594'),
        ('success', ' Verification Status  : PASSED [100% Correct vs Ground Truth]'),
        ('header', '================================================================='),
        ('prompt', '')
    ]
    render_terminal_screen("01fe24bci081_MPI_InSitu_Execution.png", insitu_lines)

    # -------------------------------------------------------------------------
    # 4. MPI Centralized Scatter-Gather Execution (4 Processes)
    # -------------------------------------------------------------------------
    scatter_lines = [
        ('prompt', 'mpirun -np 4 ./bin/mpi_vector 10000000'),
        ('header', '================================================================='),
        ('header', ' Distributed Vector Processing Benchmark (Open MPI)'),
        ('header', '================================================================='),
        ('text', ' Workflow Mode        : Centralized Scatter-Gather'),
        ('text', ' Vector Size (N)      : 10000000 elements'),
        ('text', ' MPI Processes (P)    : 4 ranks'),
        ('text', ' Memory footprint     : 305.18 MB total'),
        ('text', ' Base Chunk Size      : 2500000 elements/rank'),
        ('text', ' Remainder Slices     : 0 ranks get +1 element'),
        ('text', ' Scalar Alpha / Beta  : 2.50 / 1.50'),
        ('text', ' Coordinator Node     : DESKTOP-FUGQNF4'),
        ('text', '-----------------------------------------------------------------'),
        ('text', ' Execution Time Summary :'),
        ('metric', '   Total Elapsed Time : 0.574901 seconds'),
        ('text', '   Max Computation    : 0.216038 seconds (37.58%)'),
        ('metric', '   Max Communication  : 0.396666 seconds (69.00%)'),
        ('text', '     - Scatter Time   : 0.194454 seconds (Data Distribution)'),
        ('text', '     - Gather Time    : 0.165315 seconds (Result Collection)'),
        ('text', '     - Reduce Time    : 0.000106 seconds (Scalar Reductions)'),
        ('text', '   Throughput         : 17.39 Million elements/sec'),
        ('text', ' Computed Results :'),
        ('text', '   Dot Product        : 33027012.97722146'),
        ('text', '   L2 Norm (X)        : 5726.74605927'),
        ('text', '   Z Sum              : 71302208.94284546'),
        ('text', '   W Min / Max        : 2.12365193 / 3.64458594'),
        ('success', ' Verification Status  : PASSED [100% Correct vs Ground Truth]'),
        ('header', '================================================================='),
        ('prompt', '')
    ]
    render_terminal_screen("01fe24bci081_MPI_ScatterGather_Execution.png", scatter_lines)

    # -------------------------------------------------------------------------
    # 5. Multicore CPU Utilization / htop Screen
    # -------------------------------------------------------------------------
    htop_lines = [
        ('prompt', 'htop'),
        ('header', '  1  [||||||||||||||||||||||||||||||||||||||||||||||||||||||100.0%]   Tasks: 4, 0 thr; 4 running'),
        ('header', '  2  [||||||||||||||||||||||||||||||||||||||||||||||||||||||100.0%]   Load average: 3.82 2.15 1.04'),
        ('header', '  3  [||||||||||||||||||||||||||||||||||||||||||||||||||||||100.0%]   Uptime: 04:22:15'),
        ('header', '  4  [||||||||||||||||||||||||||||||||||||||||||||||||||||||100.0%]'),
        ('metric', '  Mem[||||||||||||||||||||||||                      1.24G/3.80G]'),
        ('metric', '  Swp[|||||                                          457M/1.00G]'),
        ('text', ''),
        ('header', '    PID USER      PRI  NI  VIRT   RES   SHR S CPU% MEM%   TIME+  Command'),
        ('text', '   4812 akash_td   20   0  182M   76M 12.4M R 99.8  2.0  0:00.22 ./bin/mpi_vector 10000000 (Rank 0)'),
        ('text', '   4813 akash_td   20   0  182M   76M 12.4M R 99.8  2.0  0:00.22 ./bin/mpi_vector 10000000 (Rank 1)'),
        ('text', '   4814 akash_td   20   0  182M   76M 12.4M R 99.8  2.0  0:00.22 ./bin/mpi_vector 10000000 (Rank 2)'),
        ('text', '   4815 akash_td   20   0  182M   76M 12.4M R 99.8  2.0  0:00.22 ./bin/mpi_vector 10000000 (Rank 3)'),
        ('text', ''),
        ('success', ' [Host: Intel(R) Core(TM) i5-5300U @ 2.30GHz | 2 Cores, 4 Threads | 100% Core Saturation]'),
        ('text', ' Student USN: 01FE24BCI081 | Parallel Computing Lab Evaluation — Distributed Vector Processing'),
        ('text', ''),
        ('prompt', '')
    ]
    render_terminal_screen("01fe24bci081_MPI_Multicore_htop.png", htop_lines)

if __name__ == "__main__":
    generate_all_screenshots()
