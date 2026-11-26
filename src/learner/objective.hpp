// ============================================================================
//  objective.hpp
// ============================================================================
//
//  Declares the objective function used by the decision-tree learner to score
//  a branch: the "Maximum Set Intersection Matching" problem.
//
//  The problem
//  -----------
//  You are given a matrix whose cells are sets of integers, and a number k.
//  Select exactly k cells such that
//
//    * no two selected cells are in the same row,
//    * no two selected cells are in the same column,
//
//  so that the size of the INTERSECTION of the k selected sets is as large as
//  possible.
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
//  It then calls solve() with k = the number of rows, so every row (leaf
//  group) is assigned to a different target. The size of the common
//  intersection is the score of the branch (the number of samples that satisfy
//  the whole assignment), and the selected cells (row, column) say which target
//  was assigned to which leaf group.
//
//  The implementation (a bitmask dynamic program) is in objective.cpp.
//
// ============================================================================

#ifndef OBJECTIVE_HPP
#define OBJECTIVE_HPP

// Provides `indexes` (a std::vector<int> assumed to be sorted, non-negative
// and free of repeated values, where `a & b` is the intersection of both) and
// `coordinates` (a list of (x, y) points). It must also make std::vector
// available, which is used below.
#include <indexes.hpp>

namespace PhantomX::Objective {

// ----------------------------------------------------------------------------
// IntSet
//
// A set of integers. It is simply `indexes`, so it is a sorted vector without
// duplicates, and the intersection of two IntSets is computed with `a & b`.
// ----------------------------------------------------------------------------
using IntSet = indexes;

// ----------------------------------------------------------------------------
// Matrix
//
// A table of sets, indexed as matrix[row][column].
//
//   * Every row must have the same number of columns (the table is
//     rectangular): the number of columns is read from the first row.
//   * It must not be empty (it must have at least one row).
// ----------------------------------------------------------------------------
using Matrix = std::vector<std::vector<IntSet>>;  // rows x columns of sets

// ----------------------------------------------------------------------------
// Result
//
// The outcome of solve().
//
// If no valid selection exists (see the preconditions of solve()), the Result
// is left in its default state: size 0 and both containers empty.
// ----------------------------------------------------------------------------
struct Result {
    // The size of the intersection of the selected sets, i.e.
    // commonElements.size(). This is the value that was maximized.
    int intersectionSize = 0;

    // The elements shared by all the selected sets (their intersection).
    IntSet commonElements;

    // The selected cells, one (x, y) pair per cell:
    //   x = the row of the cell,
    //   y = the column of the cell.
    // They are listed in increasing order of row, and no row or column
    // appears twice.
    coordinates selectedIndices;
};

// ----------------------------------------------------------------------------
// solve
//
// Selects exactly `k` cells of `matrix`, no two in the same row or column, so
// that the intersection of the selected sets is as large as possible, and
// returns the selection together with that intersection.
//
// Parameters:
//   matrix - the table of sets (see Matrix for its requirements).
//   k      - the number of cells to select.
//
// Returns:
//   The best selection found. If several selections reach the same
//   intersection size, one of them is returned (which one is unspecified).
//
// Preconditions:
//   * The matrix is not empty and is rectangular.
//   * It has at most 20 columns (the algorithm keeps one bit per column in a
//     32-bit mask and one table entry per possible mask).
//   * k <= number of rows and k <= number of columns; otherwise there is no
//     valid selection and an empty Result is returned.
//
// Cost:
//   The running time grows exponentially with the number of columns
//   (roughly 2^columns), which is why the column count is limited.
// ----------------------------------------------------------------------------
Result solve(const Matrix& matrix, int k);

} // namespace PhantomX::Objective

#endif // OBJECTIVE_HPP
