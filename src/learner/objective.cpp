// Bitmask DP for Maximum Set Intersection Matching (C++20, standard library only).
//
// Select exactly k cells, no two in the same row or column, maximizing the size
// of the intersection of the selected sets.

#include <algorithm>
#include <cstdint>

#include "objective.hpp"

namespace PhantomX::Objective {

// One DP state: the intersection so far plus the cells that produced it.
struct State {
    indexes commonElements;
    coordinates selectedIndices;
};

// ===================================================================
// Helpers: sorted-set operations
// ===================================================================

// Returns true iff every element of `small` is in `big`. O(|small| + |big|).
inline bool isSubset(const IntSet& small, const IntSet& big) {
    if (small.size() > big.size()) return false;
    std::size_t i = 0, j = 0;
    while (i < small.size()) {
        while (j < big.size() && big[j] < small[i]) ++j;
        if (j == big.size() || big[j] != small[i]) return false;
        ++i; ++j;
    }
    return true;
}

// ===================================================================
// Dominance pruning
// ===================================================================
//
// Within one mask (same set of used columns) a state with intersection X is
// dominated by one with intersection Y when X ⊆ Y: for any future set Z,
// X ∩ Z ⊆ Y ∩ Z, so Y can never end up worse. Hence each bucket is kept as an
// antichain under ⊆ (inclusion-maximal sets only). Identical sets are covered
// by X ⊆ Y with X == Y, so only one representative survives. Pruning uses true
// set inclusion, never mere cardinality.
//
// Returns true if the candidate was inserted, false if it was dominated.
inline bool insertWithDominance(std::vector<State>& bucket, State&& candidate) {
    for (const State& existing : bucket) {
        if (isSubset(candidate.commonElements, existing.commonElements)) return false;
    }
    // Candidate is not dominated: drop all existing states it dominates.
    std::erase_if(bucket, [&](const State& existing){
        return isSubset(existing.commonElements, candidate.commonElements);
    });
    bucket.push_back(std::move(candidate));
    return true;
}

// ===================================================================
// Optimized bitmask DP solver
// ===================================================================

Result solve(const Matrix& matrix, int k) {
    const std::size_t columns = matrix[0].size();
    const std::size_t rows = matrix.size();
    const std::uint32_t maskCount = std::uint32_t{1} << columns;  // columns <= 20: no overflow
    const int target = k;

    // Cache popcounts once; masks are small.
    std::vector<int> bits(maskCount);
    for (std::uint32_t m = 0; m < maskCount; ++m) bits[m] = std::popcount(m);

    // dp[mask] = antichain of non-dominated states using exactly the columns in mask.
    // Rolling layers: `current` is the table before the row, `next` after it.
    // Writing only into `next` guarantees a row contributes at most one cell.
    std::vector<std::vector<State>> current(maskCount), next(maskCount);
    current[0].push_back(State{});  // empty selection; its common set is "undefined"

    for (std::size_t row = 0; row < rows; ++row) {
        const int rowsAfter = static_cast<int>(rows - row - 1);
        for (auto& bucket : next) bucket.clear();

        for (std::uint32_t mask = 0; mask < maskCount; ++mask) {
            const int picked = bits[mask];
            if (current[mask].empty() || picked > target) continue;

            for (State& state : current[mask]) {
                // Transition B: select one cell (row, col) for each unused column.
                if (picked < target) {
                    for (std::size_t col = 0; col < columns; ++col) {
                        const std::uint32_t colBit = std::uint32_t{1} << col;
                        if (mask & colBit) continue;
                        const std::uint32_t newMask = mask | colBit;
                        // Feasibility: remaining picks must fit in remaining rows.
                        if (target - (picked + 1) > rowsAfter) continue;

                        const IntSet& cell = matrix[row][col];
                        State fresh;
                        // First selection is handled explicitly: its set IS the intersection.
                        fresh.commonElements =
                            (picked == 0) ? cell : state.commonElements & cell;
                        fresh.selectedIndices = state.selectedIndices;
                        fresh.selectedIndices.emplace_back(static_cast<int>(row),
                                                           static_cast<int>(col));
                        insertWithDominance(next[newMask], std::move(fresh));
                    }
                }
                // Transition A: skip this row; the state is moved (not copied) since
                // transition B above has already finished reading it.
                if (target - picked <= rowsAfter) {
                    if (picked == 0) {
                        // Only one "empty selection" state exists; no dominance needed.
                        if (next[mask].empty()) next[mask].push_back(std::move(state));
                    } else {
                        insertWithDominance(next[mask], std::move(state));
                    }
                }
            }
        }
        current.swap(next);
    }

    // Pick the best final state among masks with exactly k columns.
    const State* best = nullptr;
    for (std::uint32_t mask = 0; mask < maskCount; ++mask) {
        if (bits[mask] != target) continue;
        for (const State& s : current[mask]) {
            if (!best || s.commonElements.size() > best->commonElements.size()) best = &s;
        }
    }
    Result result;
    if (best) {  // always true for validated input
        result.commonElements = best->commonElements;
        result.selectedIndices = best->selectedIndices;
        result.intersectionSize = static_cast<int>(best->commonElements.size());
    }
    return result;
}

}  // namespace PhantomX