#!/usr/bin/env bash
# ==============================================================================
# Automated Benchmarking Script for Distributed Vector Processing
# Parallel & GPU Computing Lab - Team 5
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CSV_FILE="$SCRIPT_DIR/timing_results.csv"
LOG_FILE="$SCRIPT_DIR/benchmark_log.txt"

echo "======================================================================" | tee "$LOG_FILE"
echo " Starting Benchmarking Suite: Distributed Vector Processing (MPI)" | tee -a "$LOG_FILE"
echo " Host CPU Architecture: $(uname -m), $(nproc) logical cores" | tee -a "$LOG_FILE"
echo " Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

# Build binaries
cd "$ROOT_DIR"
make all

# CSV Header
echo "Paradigm,Mode,N,Processes,TotalTime,CommTime,CompTime,ScatterTime,ReduceTime,GatherTime,Throughput_M_elems_per_sec,Speedup,Efficiency,Status" > "$CSV_FILE"

SIZES=(1000000 10000000 50000000 100000000)
PROCESSES=(1 2 4 8 16)

for N in "${SIZES[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "======================================================================" | tee -a "$LOG_FILE"
    echo " WORKLOAD TIER: N = $N elements" | tee -a "$LOG_FILE"
    echo "======================================================================" | tee -a "$LOG_FILE"

    # 1. Run Sequential Baseline
    echo -n "Running Sequential Baseline (N=$N)... " | tee -a "$LOG_FILE"
    SEQ_OUT=$(./bin/sequential_vector "$N" --csv)
    echo "Done." | tee -a "$LOG_FILE"
    
    # Parse Sequential Time: Format: Sequential,N,1,exec_time,0.0,exec_time,1.0,100.0,dot_prod,z_sum
    T_SEQ=$(echo "$SEQ_OUT" | cut -d',' -f4)
    SEQ_THROUGHPUT=$(awk "BEGIN {printf \"%.2f\", $N / ($T_SEQ * 1000000)}")
    
    echo "Sequential,Sequential,$N,1,$T_SEQ,0.000000,$T_SEQ,0.000000,0.000000,0.000000,$SEQ_THROUGHPUT,1.0000,100.00,PASSED" >> "$CSV_FILE"
    echo "  -> Sequential Time: ${T_SEQ}s (Throughput: ${SEQ_THROUGHPUT} M-elem/s)" | tee -a "$LOG_FILE"

    # 2. Run In-Situ MPI Model
    for P in "${PROCESSES[@]}"; do
        echo -n "Running MPI In-Situ (N=$N, P=$P)... " | tee -a "$LOG_FILE"
        MPI_OUT=$(mpirun -np "$P" ./bin/mpi_vector "$N" --in-situ --csv)
        echo "Done." | tee -a "$LOG_FILE"

        # Format: InSitu,N,P,TotalTime,CommTime,CompTime,ScatterTime,ReduceTime,GatherTime,Throughput,DotProduct,ZSum,Status
        T_TOTAL=$(echo "$MPI_OUT" | cut -d',' -f4)
        T_COMM=$(echo "$MPI_OUT" | cut -d',' -f5)
        T_COMP=$(echo "$MPI_OUT" | cut -d',' -f6)
        T_SCAT=$(echo "$MPI_OUT" | cut -d',' -f7)
        T_RED=$(echo "$MPI_OUT" | cut -d',' -f8)
        T_GATH=$(echo "$MPI_OUT" | cut -d',' -f9)
        THROUGHPUT=$(echo "$MPI_OUT" | cut -d',' -f10)
        STATUS=$(echo "$MPI_OUT" | cut -d',' -f13)

        SPEEDUP=$(awk "BEGIN {printf \"%.4f\", $T_SEQ / $T_TOTAL}")
        EFFICIENCY=$(awk "BEGIN {printf \"%.2f\", ($T_SEQ / ($T_TOTAL * $P)) * 100}")

        echo "MPI,InSitu,$N,$P,$T_TOTAL,$T_COMM,$T_COMP,$T_SCAT,$T_RED,$T_GATH,$THROUGHPUT,$SPEEDUP,$EFFICIENCY,$STATUS" >> "$CSV_FILE"
        echo "  -> MPI InSitu (P=$P): Total=${T_TOTAL}s, Speedup=${SPEEDUP}x, Efficiency=${EFFICIENCY}%, Status=${STATUS}" | tee -a "$LOG_FILE"
    done

    # 3. Run Centralized Scatter-Gather MPI Model
    for P in "${PROCESSES[@]}"; do
        echo -n "Running MPI Scatter-Gather (N=$N, P=$P)... " | tee -a "$LOG_FILE"
        MPI_SG_OUT=$(mpirun -np "$P" ./bin/mpi_vector "$N" --csv)
        echo "Done." | tee -a "$LOG_FILE"

        T_TOTAL=$(echo "$MPI_SG_OUT" | cut -d',' -f4)
        T_COMM=$(echo "$MPI_SG_OUT" | cut -d',' -f5)
        T_COMP=$(echo "$MPI_SG_OUT" | cut -d',' -f6)
        T_SCAT=$(echo "$MPI_SG_OUT" | cut -d',' -f7)
        T_RED=$(echo "$MPI_SG_OUT" | cut -d',' -f8)
        T_GATH=$(echo "$MPI_SG_OUT" | cut -d',' -f9)
        THROUGHPUT=$(echo "$MPI_SG_OUT" | cut -d',' -f10)
        STATUS=$(echo "$MPI_SG_OUT" | cut -d',' -f13)

        SPEEDUP=$(awk "BEGIN {printf \"%.4f\", $T_SEQ / $T_TOTAL}")
        EFFICIENCY=$(awk "BEGIN {printf \"%.2f\", ($T_SEQ / ($T_TOTAL * $P)) * 100}")

        echo "MPI,ScatterGather,$N,$P,$T_TOTAL,$T_COMM,$T_COMP,$T_SCAT,$T_RED,$T_GATH,$THROUGHPUT,$SPEEDUP,$EFFICIENCY,$STATUS" >> "$CSV_FILE"
        echo "  -> MPI ScatterGather (P=$P): Total=${T_TOTAL}s, Comm=${T_COMM}s, Comp=${T_COMP}s, Speedup=${SPEEDUP}x" | tee -a "$LOG_FILE"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo " Benchmarking Suite Completed Successfully!" | tee -a "$LOG_FILE"
echo " Raw Results Saved to: $CSV_FILE" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
