// STREAM-style triad a = b + s*c (double), OpenMP, first-touch init in the same parallel layout.
// Reports best-of-10 bandwidth counting 3 arrays (2 reads + 1 write; write-allocate not counted, as in STREAM).
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    size_t n = argc > 1 ? strtoull(argv[1], 0, 10) : (1ull << 28);
    double *a = malloc(n * 8), *b = malloc(n * 8), *c = malloc(n * 8);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { a[i] = 0; b[i] = 1; c[i] = 2; }
    double best = 1e30;
    for (int r = 0; r < 10; r++) {
        double t = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
        t = omp_get_wtime() - t;
        if (t < best) best = t;
    }
    printf("{\"threads\": %d, \"n\": %zu, \"GB\": %.2f, \"best_s\": %.4f, \"triad_gbps\": %.1f, \"check\": %.1f}\n",
           omp_get_max_threads(), n, 3.0 * n * 8 / 1e9, best, 3.0 * n * 8 / best / 1e9, a[n / 2]);
    return 0;
}
