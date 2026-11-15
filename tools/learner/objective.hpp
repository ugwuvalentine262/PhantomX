#ifndef OBJECTIVE_HPP
#define OBJECTIVE_HPP

#include <indexes.hpp>

namespace PhantomX::Objective {

using IntSet = indexes;
using Matrix = std::vector<std::vector<IntSet>>;  // rows x columns of sets

struct Result {
    int intersectionSize = 0;
    std::vector<int> commonElements;
    coordinates selectedIndices;
};

Result solve(const Matrix& matrix, int k);

} // namespace PhantomX::Objective

#endif // OBJECTIVE_HPP