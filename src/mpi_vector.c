#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include "common.h"

int main(int argc, char *argv[]) {
    int rank, size;
    size_t N = 10000000; // Default: 10 Million elements
    int csv_mode = 0;
    int verify_flag = 1;
    int in_situ_mode = 0; // 0 = Scatter/Gather, 1 = In-Situ / Distributed Generation

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) {
            csv_mode = 1;
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            verify_flag = 0;
        } else if (strcmp(argv[i], "--in-situ") == 0) {
            in_situ_mode = 1;
        } else if (argv[i][0] != '-') {
            N = (size_t)atoll(argv[i]);
        }
    }

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    // Calculate domain decomposition with remainder handling
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    int base_chunk = (int)(N / (size_t)size);
    int remainder = (int)(N % (size_t)size);
    int current_disp = 0;

    for (int i = 0; i < size; i++) {
        sendcounts[i] = base_chunk + (i < remainder ? 1 : 0);
        displs[i] = current_disp;
        current_disp += sendcounts[i];
    }

    int local_n = sendcounts[rank];
    int local_offset = displs[rank];

    // Allocate local buffers on all ranks
    double *local_X = (double *)malloc((size_t)local_n * sizeof(double));
    double *local_Y = (double *)malloc((size_t)local_n * sizeof(double));
    double *local_Z = (double *)malloc((size_t)local_n * sizeof(double));
    double *local_W = (double *)malloc((size_t)local_n * sizeof(double));

    if (!local_X || !local_Y || !local_Z || !local_W) {
        fprintf(stderr, "Rank %d: Failed to allocate local memory of size %d\n", rank, local_n);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double *global_X = NULL;
    double *global_Y = NULL;
    double *global_Z = NULL;
    double *global_W = NULL;

    if (rank == 0) {
        if (!in_situ_mode || verify_flag) {
            global_X = (double *)malloc(N * sizeof(double));
            global_Y = (double *)malloc(N * sizeof(double));
            global_Z = (double *)malloc(N * sizeof(double));
            global_W = (double *)malloc(N * sizeof(double));

            if (!global_X || !global_Y || !global_Z || !global_W) {
                fprintf(stderr, "Rank 0: Failed to allocate global memory for N = %zu\n", N);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            init_vectors(global_X, global_Y, N, 0);
        }

        if (!csv_mode) {
            printf("=================================================================\n");
            printf(" Distributed Vector Processing Benchmark (Open MPI)\n");
            printf("=================================================================\n");
            printf(" Workflow Mode        : %s\n", in_situ_mode ? "In-Situ Distributed Partitioning" : "Centralized Scatter-Gather");
            printf(" Vector Size (N)      : %zu elements\n", N);
            printf(" MPI Processes (P)    : %d ranks\n", size);
            printf(" Memory footprint     : %.2f MB total\n", (double)(N * sizeof(double) * 4) / (1024.0 * 1024.0));
            printf(" Base Chunk Size      : %d elements/rank\n", base_chunk);
            printf(" Remainder Slices     : %d ranks get +1 element\n", remainder);
            printf(" Scalar Alpha / Beta  : %.2f / %.2f\n", ALPHA, BETA);
            printf(" Coordinator Node     : %s\n", hostname);
            printf("-----------------------------------------------------------------\n");
        }
    }

    // In-situ mode: each rank directly populates its own sub-domain
    if (in_situ_mode) {
        init_vectors(local_X, local_Y, (size_t)local_n, (size_t)local_offset);
    }

    // Synchronize all processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_start = MPI_Wtime();

    // -------------------------------------------------------------
    // Phase 1: Data Distribution (Scatter) [if centralized mode]
    // -------------------------------------------------------------
    double scatter_time = 0.0;
    if (!in_situ_mode) {
        double t_scatter_start = MPI_Wtime();
        MPI_Scatterv(global_X, sendcounts, displs, MPI_DOUBLE,
                     local_X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(global_Y, sendcounts, displs, MPI_DOUBLE,
                     local_Y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        double t_scatter_end = MPI_Wtime();
        scatter_time = t_scatter_end - t_scatter_start;
    }

    // -------------------------------------------------------------
    // Phase 2: Local Computation
    // -------------------------------------------------------------
    double t_comp_start = MPI_Wtime();

    double local_dot_product = 0.0;
    double local_l2_norm_sq = 0.0;
    double local_z_sum = 0.0;
    double local_w_min = 1e30;
    double local_w_max = -1e30;

    for (int i = 0; i < local_n; i++) {
        double xi = local_X[i];
        double yi = local_Y[i];

        // 1. Linear combination (SAXPY)
        double zi = ALPHA * xi + BETA * yi;
        local_Z[i] = zi;
        local_z_sum += zi;

        // 2. Non-linear mapping
        double wi = sqrt(xi * xi + yi * yi) + sin(xi) + cos(yi);
        local_W[i] = wi;
        if (wi < local_w_min) local_w_min = wi;
        if (wi > local_w_max) local_w_max = wi;

        // 3. Dot product & Norm accumulation
        local_dot_product += xi * yi;
        local_l2_norm_sq += xi * xi;
    }

    double t_comp_end = MPI_Wtime();
    double comp_time = t_comp_end - t_comp_start;

    // -------------------------------------------------------------
    // Phase 3: Global Reductions
    // -------------------------------------------------------------
    double t_red_start = MPI_Wtime();

    double global_dot_product = 0.0;
    double global_l2_norm_sq = 0.0;
    double global_z_sum = 0.0;
    double global_w_min = 0.0;
    double global_w_max = 0.0;

    MPI_Reduce(&local_dot_product, &global_dot_product, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_l2_norm_sq, &global_l2_norm_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_z_sum, &global_z_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_w_min, &global_w_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_w_max, &global_w_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double t_red_end = MPI_Wtime();
    double reduce_time = t_red_end - t_red_start;

    // -------------------------------------------------------------
    // Phase 4: Gather Computed Arrays back to Coordinator
    // -------------------------------------------------------------
    double gather_time = 0.0;
    if (!in_situ_mode) {
        double t_gather_start = MPI_Wtime();
        MPI_Gatherv(local_Z, local_n, MPI_DOUBLE,
                    global_Z, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_W, local_n, MPI_DOUBLE,
                    global_W, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        double t_gather_end = MPI_Wtime();
        gather_time = t_gather_end - t_gather_start;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_end = MPI_Wtime();
    double total_exec_time = t_total_end - t_total_start;

    // Compute max times across ranks to capture load imbalance
    double max_comp_time, max_comm_time;
    double rank_comm_time = scatter_time + reduce_time + gather_time;

    MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rank_comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // -------------------------------------------------------------
    // Verification & Reporting (Rank 0)
    // -------------------------------------------------------------
    if (rank == 0) {
        double global_l2_norm = sqrt(global_l2_norm_sq);
        int verification_passed = 1;

        if (verify_flag) {
            // Verify sequential analytical correctness
            double ref_dot = 0.0, ref_z_sum = 0.0, ref_norm_sq = 0.0;
            double ref_w_min = 1e30, ref_w_max = -1e30;

            for (size_t i = 0; i < N; i++) {
                double xi = global_X[i];
                double yi = global_Y[i];
                double zi = ALPHA * xi + BETA * yi;
                double wi = sqrt(xi * xi + yi * yi) + sin(xi) + cos(yi);

                ref_z_sum += zi;
                ref_dot += xi * yi;
                ref_norm_sq += xi * xi;
                if (wi < ref_w_min) ref_w_min = wi;
                if (wi > ref_w_max) ref_w_max = wi;

                if (!in_situ_mode) {
                    if (fabs(global_Z[i] - zi) > TOLERANCE || fabs(global_W[i] - wi) > TOLERANCE) {
                        verification_passed = 0;
                        fprintf(stderr, "Verification failed at index %zu: Z diff=%.6e, W diff=%.6e\n",
                                i, fabs(global_Z[i] - zi), fabs(global_W[i] - wi));
                        break;
                    }
                }
            }

            if (!verify_scalar(ref_dot, global_dot_product, "Dot Product") ||
                !verify_scalar(ref_z_sum, global_z_sum, "Z Sum") ||
                !verify_scalar(sqrt(ref_norm_sq), global_l2_norm, "L2 Norm")) {
                verification_passed = 0;
            }
        }

        if (!csv_mode) {
            printf(" Execution Time Summary :\n");
            printf("   Total Elapsed Time : %.6f seconds\n", total_exec_time);
            printf("   Max Computation    : %.6f seconds (%.2f%%)\n", max_comp_time, (max_comp_time / total_exec_time) * 100.0);
            printf("   Max Communication  : %.6f seconds (%.2f%%)\n", max_comm_time, (max_comm_time / total_exec_time) * 100.0);
            if (!in_situ_mode) {
                printf("     - Scatter Time   : %.6f seconds\n", scatter_time);
                printf("     - Gather Time    : %.6f seconds\n", gather_time);
            }
            printf("     - Reduce Time    : %.6f seconds\n", reduce_time);
            printf("   Throughput         : %.2f Million elements/sec\n", (double)N / (total_exec_time * 1e6));
            printf(" Computed Results :\n");
            printf("   Dot Product        : %.8f\n", global_dot_product);
            printf("   L2 Norm (X)        : %.8f\n", global_l2_norm);
            printf("   Z Sum              : %.8f\n", global_z_sum);
            printf("   W Min / Max        : %.8f / %.8f\n", global_w_min, global_w_max);
            printf(" Verification Status  : %s\n", verification_passed ? "PASSED [100% Correct]" : "FAILED");
            printf("=================================================================\n");
        } else {
            // Format: Mode,N,Processes,TotalTime,CommTime,CompTime,ScatterTime,ReduceTime,GatherTime,Throughput,DotProduct,ZSum,Status
            printf("%s,%zu,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%.8f,%.8f,%s\n",
                   in_situ_mode ? "InSitu" : "ScatterGather",
                   N, size, total_exec_time, max_comm_time, max_comp_time,
                   scatter_time, reduce_time, gather_time,
                   (double)N / (total_exec_time * 1e6),
                   global_dot_product, global_z_sum,
                   verification_passed ? "PASSED" : "FAILED");
        }

        if (global_X) free(global_X);
        if (global_Y) free(global_Y);
        if (global_Z) free(global_Z);
        if (global_W) free(global_W);
    }

    free(local_X);
    free(local_Y);
    free(local_Z);
    free(local_W);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
