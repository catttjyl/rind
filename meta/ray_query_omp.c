/* OpenMP-parallelized conservative ray/AABB traversal for the frozen c47 RayCandidateScene index.
   Bit-for-bit identical to sequential ray_query.c: each (r, t) ray writes exclusively to its own
   output row and uses thread-private binary64 entry distance buffer. */
#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline int slab(const double *origin, const double *direction,
                       const double *low, const double *high, double horizon, double *entry) {
    double enter = 0.0, leave = horizon;
    for (int axis = 0; axis < 3; ++axis) {
        double q = origin[axis], d = direction[axis];
        if (d == 0.0) {
            if (q < low[axis] || q > high[axis]) return 0;
            continue;
        }
        double a = (low[axis] - q) / d, b = (high[axis] - q) / d;
        double near = a < b ? a : b, far = a < b ? b : a;
        if (near > enter) enter = near;
        if (far < leave) leave = far;
        if (enter > leave) return 0;
    }
    *entry = enter;
    return 1;
}

int query_rays(int64_t nr, int64_t nt, int64_t capacity, int64_t root, double horizon,
               const double *origins, const double *directions, const uint64_t *own,
               const double *low, const double *high, const int64_t *links,
               const int64_t *order, const double *face_low, const double *face_high,
               const uint8_t *reasons, const uint64_t *refs,
               int64_t *selected, int64_t *counts, double *entries_unused) {
    (void)entries_unused;
    if (root < 0 || nr <= 0 || nt <= 0) return 0;
    int64_t total = nr * nt;
    int global_err = 0;

    #pragma omp parallel
    {
        double stack_entries[256];
        double *local_entries = (capacity <= 256)
            ? stack_entries
            : (double *)malloc((size_t)capacity * sizeof(double));
        if (!local_entries) {
            #pragma omp critical
            {
                int cur = __atomic_load_n(&global_err, __ATOMIC_ACQUIRE);
                if (cur == 0) __atomic_store_n(&global_err, 3, __ATOMIC_RELEASE);
            }
        }
        /* All threads MUST encounter the worksharing loop to satisfy OpenMP barrier semantics. */
        #pragma omp for schedule(dynamic, 32)
        for (int64_t idx = 0; idx < total; ++idx) {
            if (!local_entries || __atomic_load_n(&global_err, __ATOMIC_ACQUIRE) != 0) continue;
            int64_t r = idx / nt;
            int64_t t = idx % nt;
            const double *q = origins + 3 * r;
            const double *d = directions + 3 * t;
            int64_t stack[64], stack_size = 1, count = 0;
            stack[0] = root;
            int64_t *row = selected + idx * capacity;
            int local_err = 0;
            while (stack_size > 0) {
                int64_t node = stack[--stack_size];
                double entry;
                if (!slab(q, d, low + 3 * node, high + 3 * node, horizon, &entry)) continue;
                const int64_t *link = links + 4 * node;
                if (link[0] >= 0) {
                    if (stack_size + 2 > 64) { local_err = 2; break; }
                    stack[stack_size++] = link[1];
                    stack[stack_size++] = link[0];
                    continue;
                }
                for (int64_t j = link[2]; j < link[3]; ++j) {
                    int64_t face = order[j];
                    if (!slab(q, d, face_low + 3 * face, face_high + 3 * face, horizon, &entry)) continue;
                    if (reasons[face] == 2) { local_err = 1; break; }
                    if (own && refs[face] == own[r]) continue;
                    int64_t slot = count < capacity ? count : capacity;
                    ++count;
                    while (slot > 0) {
                        int64_t prev = slot - 1;
                        if (local_entries[prev] < entry ||
                            (local_entries[prev] == entry && refs[row[prev]] < refs[face])) break;
                        if (slot < capacity) {
                            local_entries[slot] = local_entries[prev];
                            row[slot] = row[prev];
                        }
                        slot = prev;
                    }
                    if (slot < capacity) {
                        local_entries[slot] = entry;
                        row[slot] = face;
                    }
                }
                if (local_err != 0) break;
            }
            if (local_err != 0) {
                #pragma omp critical
                {
                    int cur = __atomic_load_n(&global_err, __ATOMIC_ACQUIRE);
                    if (cur == 0 || local_err < cur) __atomic_store_n(&global_err, local_err, __ATOMIC_RELEASE);
                }
            } else {
                counts[idx] = count;
            }
        }
        if (capacity > 256 && local_entries) free(local_entries);
    }
    return __atomic_load_n(&global_err, __ATOMIC_ACQUIRE);
}
