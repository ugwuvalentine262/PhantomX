#ifndef OBJECTIVE_HPP
#define OBJECTIVE_HPP

#include <vector>

namespace PhantomX::Objective {

using IntSet = std::vector<int>;                  // sorted ascending, distinct, nonnegative
using Matrix = std::vector<std::vector<IntSet>>;  // rows x columns of sets

struct coordinate { int row, col; };              // (row, column), zero-indexed

struct Result {
    int intersectionSize = 0;
    std::vector<int> commonElements;
    std::vector<coordinate> selectedIndices;
};

Result solve(const Matrix& matrix, int k);

} // namespace PhantomX::Objective

#endif // OBJECTIVE_HPP