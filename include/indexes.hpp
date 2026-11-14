#ifndef INDEXES_HPP
#define INDEXES_HPP

#include <vector>

namespace PhantomX {

struct coordinate { int x, y; };

using coordinates = std::vector<coordinate>;
using indexes = std::vector<int>; // sorted ascending, distinct, nonnegative

// Returns a new set of indexes formed from the intersection
// of this object and the indexes specified as arguments to this function.
// Two-pointer intersection, O(|a| + |b|). Inputs sorted & distinct => output too.
inline indexes operator&(const indexes& _a, const indexes& _b)
{
    indexes out;
    out.reserve(std::min(_a.size(), _b.size()));
    std::size_t i = 0, j = 0;
    while (i < _a.size() && j < _b.size()) {
        if (_a[i] < _b[j]) ++i;
        else if (_b[j] < _a[i]) ++j;
        else { out.push_back(_a[i]); ++i; ++j; }
    }
    return out;
}

} // namespace PhantomX

#endif // INDEXES_HPP