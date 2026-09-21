#ifndef _STRATEGY_H
#define _STRATEGY_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Splits for an optimal isogeny-chain strategy. With m levels left, descend
 * split[m] of them by doubling and keep the remaining m - split[m] to be pushed
 * through the isogenies below. The balanced split[m] = m/2 is optimal only when
 * a doubling and an evaluation cost the same; p and q are the per-level costs of
 * each, in any matching unit.
 */
static inline void
strategy_splits(uint16_t *split, const int n, const uint32_t p, const uint32_t q)
{
    uint64_t cost[n + 1];

    cost[1] = 0;
    split[1] = 0;
    for (int m = 2; m <= n; m++) {
        uint64_t best = UINT64_MAX;
        int best_i = 1;
        for (int i = 1; i < m; i++) {
            uint64_t c = cost[i] + cost[m - i] + (uint64_t)i * p + (uint64_t)(m - i) * q;
            if (c < best) {
                best = c;
                best_i = i;
            }
        }
        cost[m] = best;
        split[m] = (uint16_t)best_i;
    }
}

/*
 * Stack the chain traversal needs for these splits: the longest run of "descend
 * and keep" steps, m -> m - split[m] -> ... -> 1, plus a slot of margin. The
 * 1 + ceil(log2 m) used for a balanced strategy does not cover a skewed one.
 */
static inline int
strategy_space(const uint16_t *split, int m)
{
    int space = 2;

    while (m > 1) {
        m -= split[m];
        space++;
    }
    return space;
}

#endif
