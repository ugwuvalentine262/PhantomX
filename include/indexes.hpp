// ============================================================================
//  indexes.hpp
// ============================================================================
//
//  Declares the small geometric/indexing primitives shared across the
//  PhantomX library:
//
//    - `coordinate` and `coordinates`: a 2-D integer coordinate and a
//      vector of them. Used by `decision_tree::leaf` to describe how a
//      target value is derived.
//
//    - `indexes`: a set of integer indices, stored as a vector sorted
//      ascending and free of repeats.
//
//    - `operator&`: set intersection on `indexes`, exploiting the sorted
//      and distinct invariant.
//
//  Invariant on `indexes`:
//
//    Throughout the library, an `indexes` value is expected to be sorted
//    ascending, contain no repeats, and contain only non-negative values.
//    This invariant is what makes `operator&` a linear-time two-pointer
//    merge, and what lets `decision_tree::learner`'s `state` and `group`
//    logic reason about subsets cheaply. It is NOT enforced by the type
//    itself — callers must maintain it.
//
// ============================================================================

#ifndef INDEXES_HPP
#define INDEXES_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

namespace PhantomX {

// ----------------------------------------------------------------------------
// coordinate
//
// A single 2-D integer coordinate. Used as the element type of
// `coordinates`.
// ----------------------------------------------------------------------------
struct coordinate {
    int x, y;
};

// ----------------------------------------------------------------------------
// coordinates
//
// A vector of coordinates. In `decision_tree::leaf`, this describes how the
// target value is derived at that leaf.
// ----------------------------------------------------------------------------
using coordinates = std::vector<coordinate>;

// ----------------------------------------------------------------------------
// indexes
//
// A set of integer indices, stored as a vector.
//
// Invariant: sorted ascending, distinct, non-negative, non-repeating.
// Maintained by callers; not enforced by the type. The sorted-and-distinct
// property is relied upon by `operator&` and by the decision-tree learner.
// ----------------------------------------------------------------------------
using indexes = std::vector<int>; // sorted, ascending, distinct, nonnegative, nonrepeating

// ----------------------------------------------------------------------------
// operator&
//
// Returns a new set of indexes formed from the intersection of this object
// and the indexes specified as arguments to this function.
//
// Two-pointer intersection, O(|a| + |b|). Inputs sorted & distinct => output
// too. Exploits the `indexes` invariant: because both inputs are sorted and
// distinct, a single forward pass suffices, and the output inherits the
// same sorted-and-distinct property.
//
// NOTE: this is an overload of the bitwise-AND operator on `indexes`. It is
// used deliberately as set-intersection sugar (e.g. `g1.indexes & g2.indexes`
// in `decision_tree::learner::group`'s constructor). It is not bitwise AND.
// ----------------------------------------------------------------------------
inline indexes operator&(const indexes& _a, const indexes& _b)
{
    indexes result;
    // The intersection cannot be larger than the smaller of the two inputs,
    // so reserve that much to avoid reallocation.
    result.reserve(std::min(_a.size(), _b.size()));

    std::size_t i = 0, j = 0;
    while (i < _a.size() && j < _b.size()) {
        if (_a[i] < _b[j]) {
            ++i;
        } else if (_b[j] < _a[i]) {
            ++j;
        } else {
            // _a[i] == _b[j]: common element, emit and advance both.
            result.push_back(_a[i]);
            ++i;
            ++j;
        }
    }
    return result;
}

} // namespace PhantomX

#endif // INDEXES_HPP