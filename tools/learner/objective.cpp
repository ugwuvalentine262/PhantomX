// Bitmask DP for Maximum Set Intersection Matching (C++20, standard library only).
//
// Select exactly k cells, no two in the same row or column, maximizing the size
// of the intersection of the selected sets.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "objective.hpp"

namespace PhantomX::Objective {

// One DP state: the intersection so far plus the cells that produced it.
struct State {
    std::vector<int> commonElements;
    std::vector<coordinate> selectedIndices;
};

// Columns are tracked in a bitmask; the DP table has 2^m buckets, so cap m.
constexpr std::size_t kMaxColumns = 20;

// ===================================================================
// Helpers: sorted-set operations
// ===================================================================

// Two-pointer intersection, O(|a| + |b|). Inputs sorted & distinct => output too.
inline IntSet intersectSorted(const IntSet& a, const IntSet& b) {
    IntSet out;
    out.reserve(std::min(a.size(), b.size()));
    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) ++i;
        else if (b[j] < a[i]) ++j;
        else { out.push_back(a[i]); ++i; ++j; }
    }
    return out;
}

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
// Input validation
// ===================================================================

// Returns the column count. Throws std::invalid_argument on malformed input.
inline std::size_t validateInput(const Matrix& matrix, int k) {
    if (matrix.empty()) throw std::invalid_argument("matrix has no rows");
    const std::size_t columns = matrix[0].size();
    if (columns == 0) throw std::invalid_argument("matrix has no columns");
    if (columns > kMaxColumns)
        throw std::invalid_argument("too many columns for bitmask DP (max " +
                                    std::to_string(kMaxColumns) + ")");
    for (std::size_t r = 0; r < matrix.size(); ++r) {
        if (matrix[r].size() != columns)
            throw std::invalid_argument("ragged matrix: row " + std::to_string(r));
        for (std::size_t c = 0; c < columns; ++c) {
            const IntSet& s = matrix[r][c];
            for (std::size_t i = 0; i < s.size(); ++i) {
                if (s[i] < 0) throw std::invalid_argument("negative element");
                if (i > 0 && s[i] == s[i - 1]) throw std::invalid_argument("duplicate element in set");
                if (i > 0 && s[i] < s[i - 1]) throw std::invalid_argument("set is not sorted ascending");
            }
        }
    }
    if (k < 1) throw std::invalid_argument("k must be >= 1");
    if (static_cast<std::size_t>(k) > std::min(matrix.size(), columns))
        throw std::invalid_argument("k exceeds min(rows, columns)");
    return columns;
}

// ===================================================================
// Optimized bitmask DP solver
// ===================================================================

inline Result solve(const Matrix& matrix, int k) {
    const std::size_t columns = validateInput(matrix, k);
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
                            (picked == 0) ? cell : intersectSorted(state.commonElements, cell);
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

// ===================================================================
// Brute-force reference solver (independent oracle; small inputs only)
// ===================================================================

namespace brute {

inline void enumerate(const Matrix& matrix, int k, std::size_t row, std::uint32_t usedColumns,
                      int picked, std::vector<coordinate>& chosen, Result& best, bool& haveBest) {
    const int rowsLeft = static_cast<int>(matrix.size() - row);
    if (picked + rowsLeft < k) return;
    if (picked == k) {
        // Recompute the intersection from scratch with a naive membership test.
        IntSet common = matrix[chosen[0].row][chosen[0].col];
        for (std::size_t i = 1; i < chosen.size(); ++i) {
            const IntSet& other = matrix[chosen[i].row][chosen[i].col];
            IntSet kept;
            for (int v : common)
                if (std::find(other.begin(), other.end(), v) != other.end()) kept.push_back(v);
            common = std::move(kept);
        }
        if (!haveBest || static_cast<int>(common.size()) > best.intersectionSize) {
            haveBest = true;
            best.intersectionSize = static_cast<int>(common.size());
            best.commonElements = common;
            best.selectedIndices = chosen;
        }
        return;
    }
    if (row == matrix.size()) return;
    enumerate(matrix, k, row + 1, usedColumns, picked, chosen, best, haveBest);  // skip row
    for (std::size_t col = 0; col < matrix[row].size(); ++col) {
        if (usedColumns & (std::uint32_t{1} << col)) continue;
        chosen.emplace_back(static_cast<int>(row), static_cast<int>(col));
        enumerate(matrix, k, row + 1, usedColumns | (std::uint32_t{1} << col), picked + 1, chosen,
                  best, haveBest);
        chosen.pop_back();
    }
}

Result solve(const Matrix& matrix, int k) {
    Result best;
    bool haveBest = false;
    std::vector<coordinate> chosen;
    enumerate(matrix, k, 0, 0, 0, chosen, best, haveBest);
    return best;
}

}  // namespace brute

}  // namespace PhantomX