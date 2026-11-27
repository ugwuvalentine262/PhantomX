// ============================================================================
//  objective.hpp
// ============================================================================
//
//  Declares the objective function used by the decision-tree learner to score
//  a branch: the "Maximum-Weight Set Intersection Matching" problem.
//
//  The problem
//  -----------
//  You are given a matrix whose cells contain weighted sample records, and a
//  number k.
//  Select exactly k cells such that
//
//    * no two selected cells are in the same row,
//    * no two selected cells are in the same column,
//
//  so that the sum of the weights in the intersection of the k selected sample
//  sets is as large as possible. The resulting common set retains each
//  member's sample weight.
//
//  How the learner uses it
//  -----------------------
//  decision_tree::learner::divide() builds one matrix per branch of a
//  candidate split:
//
//      row    = one leaf group of the decision path,
//      column = one target,
//      cell   = the samples that belong to that group and that target.
//
//  It then calls solve() with a k chosen by the learner. Each selected cell
//  assigns a different target to a different leaf-group row. The common sample
//  records are used to calculate the branch's weighted score, and the selected
//  cells (row, column) describe the assignment.
//
//  The implementation (a bitmask dynamic program) is in objective.cpp.
//
// ============================================================================

#ifndef OBJECTIVE_HPP
#define OBJECTIVE_HPP

// Provides `indexes` and `coordinates`, and makes the standard containers used
// by the objective types available.
#include <indexes.hpp>

#include <cstddef>  // std::size_t
#include <vector>   // std::vector
#include <numeric>  // std::accumulate

namespace PhantomX::Objective {

// ============================================================================
//  element
// ============================================================================
// One weighted sample record. The sample index is used to identify the record
// and to order the records in a set, so that two sets can be intersected with.
struct element {
    int sample; // the sample index
    double score; // the weight of the sample
};

// ----------------------------------------------------------------------------
// set
//
// A weighted set of samples. Records are ordered by sample index so the
// objective can intersect two sets with a linear two-pointer scan.
// ----------------------------------------------------------------------------
using set = std::vector<element>;

// ----------------------------------------------------------------------------
// matrix
//
// A table of sets, indexed as matrix[row][column].
//
//   * Every row must have the same number of columns (the table is
//     rectangular): the number of columns is read from the first row.
//   * It must not be empty (it must have at least one row).
// ----------------------------------------------------------------------------
using matrix = std::vector<std::vector<set>>;  // rows x columns of sets

// ----------------------------------------------------------------------------
// result
//
// The outcome of solve().
//
// If no valid selection exists (see the preconditions of solve()), the result
// is left in its default state: `k` is zero and both containers are empty.
// ----------------------------------------------------------------------------
struct result {
    // The weighted sample records shared by all selected cells.
    set common_elements;

    // The selected cells, one (x, y) pair per cell:
    //   x = the row of the cell,
    //   y = the column of the cell.
    // They are listed in increasing order of row, and no row or column
    // appears twice.
    coordinates selected_indices;

    // parameter used for solving.
    std::size_t k = 0;

public:
    // Returns the sum of the record weights in a set. O(n) in the set size.
    static inline double weighted_sum(const set& set) {
        return std::accumulate(set.begin(), set.end(), 0.0,
            [](double sum, const element& e) { return sum + e.score; });
    }
};

// ----------------------------------------------------------------------------
// solve
//
// Selects exactly `k` cells of `matrix`, no two in the same row or column, so
// that the sum of the weights in the intersection of the selected sets is as
// large as possible, and returns the selection together with that weighted
// intersection.
//
// Parameters:
//   matrix - the table of weighted sample sets (see matrix for its requirements).
//   k      - the number of cells to select.
//
// Returns:
//   The best selection found. If several selections reach the same
//   total intersection weight, one of them is returned (which one is
//   unspecified).
//
// Preconditions:
//   * The matrix is not empty and is rectangular.
//   * It has at most 20 columns (the algorithm keeps one bit per column in a
//     32-bit mask and one table entry per possible mask).
//   * k <= number of rows and k <= number of columns; otherwise there is no
//     valid selection and an empty result is returned.
//
// Cost:
//   The running time grows exponentially with the number of columns
//   (roughly 2^columns), which is why the column count is limited.
// ----------------------------------------------------------------------------
result solve(const matrix& matrix, int k);

} // namespace PhantomX::Objective

#endif // OBJECTIVE_HPP
