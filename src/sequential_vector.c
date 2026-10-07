#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"

int main(int argc, char *argv[]) {
    size_t N = 10000000; // Default: 10 Million elements
    int csv_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) {
            csv_mode = 1;
        } else if (argv[i][0] != '-') {
            N = (size_t)atoll(argv[i]);
        }
    }

    if (!csv_mode) {
        printf("=================================================================\n");
        printf(" Sequential Vector Processing Benchmark (Baseline)\n");
        printf("=================================================================\n");
        printf(" Vector Size (N)      : %zu elements\n", N);
        printf(" Memory footprint (X, Y, Z, W) : %.2f MB\n", (double)(N * sizeof(double) * 4) / (1024.0 * 1024.0));
        printf(" Scalar Alpha / Beta  : %.2f / %.2f\n", ALPHA, BETA);
        printf("-----------------------------------------------------------------\n");
    }

    double *X = (double *)malloc(N * sizeof(double));
    double *Y = (double *)malloc(N * sizeof(double));
    double *Z = (double *)malloc(N * sizeof(double));
    double *W = (double *)malloc(N * sizeof(double));

    if (!X || !Y || !Z || !W) {
        fprintf(stderr, "Error: Memory allocation failed for size %zu\n", N);
        return 1;
    }

    // Initialize vectors deterministically
    init_vectors(X, Y, N, 0);

    double t_start = get_wall_time();

    double dot_product = 0.0;
    double l2_norm_sq = 0.0;
    double z_sum = 0.0;
    double w_min = 1e30;
    double w_max = -1e30;

    // Vector operations pipeline
    for (size_t i = 0; i < N; i++) {
        double xi = X[i];
        double yi = Y[i];

        // 1. Linear combination (SAXPY-like)
        double zi = ALPHA * xi + BETA * yi;
        Z[i] = zi;
        z_sum += zi;

        // 2. Non-linear mathematical pipeline
        double wi = sqrt(xi * xi + yi * yi) + sin(xi) + cos(yi);
        W[i] = wi;
        if (wi < w_min) w_min = wi;
        if (wi > w_max) w_max = wi;

        // 3. Dot product & Norm accumulation
        dot_product += xi * yi;
        l2_norm_sq += xi * xi;
    }

    double l2_norm = sqrt(l2_norm_sq);

    double t_end = get_wall_time();
    double exec_time = t_end - t_start;

    if (!csv_mode) {
        printf(" Execution Time       : %.6f seconds\n", exec_time);
        printf(" Throughput           : %.2f Million elements/sec\n", (double)N / (exec_time * 1e6));
        printf(" Verification Samples :\n");
        printf("   Z[0] = %.8f, Z[N-1] = %.8f\n", Z[0], Z[N - 1]);
        printf("   W[0] = %.8f, W[N-1] = %.8f\n", W[0], W[N - 1]);
        printf("   Dot Product        : %.8f\n", dot_product);
        printf("   L2 Norm (X)        : %.8f\n", l2_norm);
        printf("   Z Sum              : %.8f\n", z_sum);
        printf("   W Min / Max        : %.8f / %.8f\n", w_min, w_max);
        printf("=================================================================\n");
    } else {
        // Format: paradigm,N,processes,exec_time,comm_time,comp_time,speedup,efficiency,dot_product,z_sum
        printf("Sequential,%zu,1,%.6f,0.000000,%.6f,1.0000,100.00,%.6f,%.6f\n",
               N, exec_time, exec_time, dot_product, z_sum);
    }

    free(X);
    free(Y);
    free(Z);
    free(W);

    return 0;
}
