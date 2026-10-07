#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

#define ALPHA 2.5
#define BETA  1.5
#define TOLERANCE 1e-6

// High-resolution wall-clock timer for sequential benchmarking
static inline double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Deterministic vector initialization
static inline void init_vectors(double *X, double *Y, size_t n, size_t offset) {
    for (size_t i = 0; i < n; i++) {
        size_t global_idx = offset + i;
        // Use normalized trigonometric functions to avoid overflow while maintaining non-trivial computation
        X[i] = sin((double)(global_idx % 1000) * 0.01) + 1.5;
        Y[i] = cos((double)(global_idx % 1000) * 0.01) + 2.0;
    }
}

// Verification function comparing test array with reference array
static inline int verify_results(const double *ref, const double *test, size_t n, const char *label) {
    double max_diff = 0.0;
    size_t err_idx = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = fabs(ref[i] - test[i]);
        if (diff > max_diff) {
            max_diff = diff;
            err_idx = i;
        }
    }
    if (max_diff > TOLERANCE) {
        fprintf(stderr, "Verification FAILED for %s at index %zu: ref=%.8f, test=%.8f (diff=%.8e)\n",
                label, err_idx, ref[err_idx], test[err_idx], max_diff);
        return 0;
    }
    return 1;
}

// Verification function for scalar reduction values (e.g. dot product, sum)
static inline int verify_scalar(double ref, double test, const char *label) {
    double diff = fabs(ref - test);
    double rel_err = (fabs(ref) > 1e-12) ? (diff / fabs(ref)) : diff;
    if (rel_err > TOLERANCE) {
        fprintf(stderr, "Verification FAILED for %s: ref=%.8f, test=%.8f (rel_err=%.8e)\n",
                label, ref, test, rel_err);
        return 0;
    }
    return 1;
}

#endif // COMMON_H
