// Bitmask DP for Maximum-Weight Set Intersection Matching (C++20, standard
// library only).
//
// Select exactly k cells, no two in the same row or column, maximizing the sum
// of the weights in the intersection of the selected sets.

#include <algorithm>
#include <cstdint>

#include "objective.hpp"

namespace PhantomX::Objective {

// One DP state: the intersection so far plus the cells that produced it.
struct state {
    set common_elements;
    coordinates selected_indices;
};

namespace {
// ===================================================================
// Helpers: sorted-set operations
// ===================================================================

// Returns true iff every element of `small` is in `big`. O(|small| + |big|).
bool isSubset(const set& small, const set& big) {
    if (small.size() > big.size()) return false;
    std::size_t i = 0, j = 0;
    while (i < small.size()) {
        while (j < big.size() && big[j].sample < small[i].sample) ++j;
        if (j == big.size() || big[j].sample != small[i].sample) return false;
        ++i; ++j;
    }
    return true;
}

// Replaces `result` with the weighted intersection of the two specified sets
// and returns `result`.
//
// Two-pointer intersection, O(|a| + |b|). Inputs sorted & distinct => output
// too. Exploits the `indexes` invariant: because both inputs are sorted and
// distinct, a single forward pass suffices, and the output inherits the
// same sorted-and-distinct property.
set& Intersect(const set& a, const set& b, set& result) {

    result.clear();
    // The intersection cannot contain more records than the smaller input,
    // so reserve that much to avoid reallocation.
    result.reserve(std::min(a.size(), b.size()));

    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i].sample < b[j].sample) {
            ++i;
        } else if (b[j].sample < a[i].sample) {
            ++j;
        } else {
            // a[i] == b[j]: common element, emit and advance both.
            result.push_back(a[i]);
            ++i, ++j;
        }
    }
    return result;
}

// ===================================================================
// Dominance pruning
// ===================================================================
//
// Because sample weights are non-negative, a superset intersection always has
// at least as much total weight as its subset. Inclusion-based dominance is
// therefore valid for the weighted objective as well as for cardinality.
//
// Within one mask (same set of used columns) a state with intersection X is
// dominated by one with intersection Y when X ⊆ Y: for any future set Z,
// X ∩ Z ⊆ Y ∩ Z, so Y can never end up worse. Hence each bucket is kept as an
// antichain under ⊆ (inclusion-maximal sets only). Identical sets are covered
// by X ⊆ Y with X == Y, so only one representative survives. Pruning uses true
// set inclusion; cardinality alone is not used for pruning.
//
// Returns true if the candidate was inserted, false if it was dominated.
bool insertWithDominance(std::vector<state>& bucket, state&& candidate) {
    for (const state& existing : bucket) {
        if (isSubset(candidate.common_elements, existing.common_elements)) return false;
    }
    // Candidate is not dominated: drop all existing states it dominates.
    std::erase_if(bucket, [&](const state& existing){
        return isSubset(existing.common_elements, candidate.common_elements);
    });
    bucket.push_back(std::move(candidate));
    return true;
}

} // anonymous namespace

// ===================================================================
// Optimized bitmask DP solver
// ===================================================================

result solve(const matrix& matrix, int k) {
    const std::size_t columns = matrix[0].size();
    const std::size_t rows = matrix.size();
    const std::uint32_t maskCount = std::uint32_t{1} << columns;  // columns <= 20: no overflow
    const int target = k;

    set intersect;

    // Cache popcounts once; masks are small.
    std::vector<int> bits(maskCount);
    for (std::uint32_t m = 0; m < maskCount; ++m) bits[m] = std::popcount(m);

    // dp[mask] = antichain of non-dominated states using exactly the columns in mask.
    // Rolling layers: `current` is the table before the row, `next` after it.
    // Writing only into `next` guarantees a row contributes at most one cell.
    std::vector<std::vector<state>> current(maskCount), next(maskCount);
    current[0].push_back(state{});  // empty selection; its common set is "undefined"

    for (std::size_t row = 0; row < rows; ++row) {
        const int rowsAfter = static_cast<int>(rows - row - 1);
        for (auto& bucket : next) bucket.clear();

        for (std::uint32_t mask = 0; mask < maskCount; ++mask) {
            const int picked = bits[mask];
            if (current[mask].empty() || picked > target) continue;

            for (state& st : current[mask]) {
                // Transition B: select one cell (row, col) for each unused column.
                if (picked < target) {
                    for (std::size_t col = 0; col < columns; ++col) {
                        const std::uint32_t colBit = std::uint32_t{1} << col;
                        if (mask & colBit) continue;
                        const std::uint32_t newMask = mask | colBit;
                        // Feasibility: remaining picks must fit in remaining rows.
                        if (target - (picked + 1) > rowsAfter) continue;

                        const set& cell = matrix[row][col];
                        state fresh;
                        // First selection is handled explicitly: its set IS the intersection.
                        fresh.common_elements =
                            (picked == 0) ? cell : Intersect(st.common_elements, cell, intersect);
                        fresh.selected_indices = st.selected_indices;
                        fresh.selected_indices.emplace_back(static_cast<int>(row),
                                                           static_cast<int>(col));
                        insertWithDominance(next[newMask], std::move(fresh));
                    }
                }
                // Transition A: skip this row; the state is moved (not copied) since
                // transition B above has already finished reading it.
                if (target - picked <= rowsAfter) {
                    if (picked == 0) {
                        // Only one "empty selection" state exists; no dominance needed.
                        if (next[mask].empty()) next[mask].push_back(std::move(st));
                    } else {
                        insertWithDominance(next[mask], std::move(st));
                    }
                }
            }
        }
        current.swap(next);
    }

    // Pick the final state with the greatest total weight in its common set
    // among masks with exactly k columns.
    const state* best = nullptr;
    for (std::uint32_t mask = 0; mask < maskCount; ++mask) {
        if (bits[mask] != target) continue;
        for (const state& s : current[mask]) {
            if (!best ||
                result::weighted_sum(s.common_elements) >
                    result::weighted_sum(best->common_elements)) {
                best = &s;
            }
        }
    }
    result result;
    if (best) {  // always true for validated input
        result.common_elements = best->common_elements;
        result.selected_indices = best->selected_indices;
        result.k = k;
    }
    return result;
}

}  // namespace PhantomX
