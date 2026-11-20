/* Conservative ray/AABB traversal for the frozen c47 RayCandidateScene index.
   Inputs are immutable. Distances use binary64 and ties use uint64 source IDs. */
#include <stdint.h>

static int slab(const double *origin, const double *direction,
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
               int64_t *selected, int64_t *counts, double *entries) {
    if (root < 0) return 0;
    for (int64_t r = 0; r < nr; ++r) {
        for (int64_t t = 0; t < nt; ++t) {
            const double *q = origins + 3*r, *d = directions + 3*t;
            int64_t stack[64], stack_size = 1, count = 0;
            stack[0] = root;
            int64_t *row = selected + (r*nt + t)*capacity;
            while (stack_size > 0) {
                int64_t node = stack[--stack_size];
                double entry;
                if (!slab(q, d, low + 3*node, high + 3*node, horizon, &entry)) continue;
                const int64_t *link = links + 4*node;
                if (link[0] >= 0) {
                    if (stack_size + 2 > 64) return 2;
                    stack[stack_size++] = link[1];
                    stack[stack_size++] = link[0];
                    continue;
                }
                for (int64_t j = link[2]; j < link[3]; ++j) {
                    int64_t face = order[j];
                    if (!slab(q, d, face_low + 3*face, face_high + 3*face, horizon, &entry)) continue;
                    if (reasons[face] == 2) return 1;
                    if (own && refs[face] == own[r]) continue;
                    int64_t slot = count < capacity ? count : capacity;
                    ++count;
                    while (slot > 0) {
                        int64_t prev = slot - 1;
                        if (entries[prev] < entry ||
                            (entries[prev] == entry && refs[row[prev]] < refs[face])) break;
                        if (slot < capacity) {
                            entries[slot] = entries[prev];
                            row[slot] = row[prev];
                        }
                        slot = prev;
                    }
                    if (slot < capacity) {
                        entries[slot] = entry;
                        row[slot] = face;
                    }
                }
            }
            counts[r*nt + t] = count;
        }
    }
    return 0;
}
