// ============================================================================
//  dataset.hpp
// ============================================================================
//
//  Declares `dataset`, the in-memory container for training samples and
//  their targets.
//
//  Role of this type:
//
//    - `dataset` holds a flat vector of `sample`s, each carrying the
//      attribute indices that describe the sample ("x") and the packed
//      target values ("y").
//
//    - It is loaded once from a stream against a `configuration`, which
//      tells it how attribute/label indices are grouped. It then acts as
//      read-only reference data for learners and thinkers.
//
//    - The class stores attribute *indices*, not attribute values: the
//      actual feature values live elsewhere and are looked up by index.
//
//  Sample representation:
//
//    Each `sample` has two parallel parts:
//
//      - `x`: the attribute indices that describe this sample.
//      - `y`: a fixed-size array of up to eight 16-bit target codes, one
//        per target goal (see `config::id()`). A slot left at its sentinel
//        value `0xffff` means "no target for this goal".
//
// ============================================================================

#ifndef DATASET_HPP
#define DATASET_HPP

#include <cstddef>
#include <cstdint>
#include <istream>
#include <vector>

#include <configuration.hpp>
#include <indexes.hpp>

namespace PhantomX {

// ----------------------------------------------------------------------------
// dataset
//
// The in-memory collection of training samples plus their targets. Loading
// is delegated to the constructor; everything afterwards is read-only
// access.
// ----------------------------------------------------------------------------
struct dataset {

    // ------------------------------------------------------------------------
    // sample
    //
    // A single training example: the attribute indices that describe it
    // plus its (up to eight) packed target codes.
    // ------------------------------------------------------------------------
    struct sample {

        // The attribute indices describing this sample.
        indexes x;

        // The target codes, one per goal. A slot left at the sentinel
        // value `0xffff` means "no target for this goal".
        //
        // The array is fixed at eight slots, so at most eight goals are
        // supported per sample. Each code is a `config::id()` result (a
        // 16-bit index into the configuration's sorted "y"), with `0xffff`
        // reserved as the "absent" sentinel — which is why `id()` itself
        // returns `0xffff` for a missing key.
        uint16_t y[8] = {
            0xffff, 0xffff, 0xffff, 0xffff,
            0xffff, 0xffff, 0xffff, 0xffff
        };
    };

    // ------------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------------

    // The samples that make up the dataset.
    std::vector<sample> samples;

    // True iff the dataset was loaded successfully by the constructor.
    // Mirrored by `operator bool()`.
    bool okay = false;

public:

    // ------------------------------------------------------------------------
    // Validity
    // ------------------------------------------------------------------------

    // Boolean type-cast operator. Returns true iff the dataset was loaded
    // successfully by the constructor. Equivalent to reading `okay`.
    operator bool() const;

    // ------------------------------------------------------------------------
    // Access
    // ------------------------------------------------------------------------

    // Index operator. Returns a reference to the sample at the specified
    // index. The index is not bounds-checked here; out-of-range access is
    // undefined behaviour.
    const sample& operator[](std::size_t _index) const;

    // Gets the number of samples in the dataset.
    size_t size() const;

    // ------------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------------

    // Constructor.
    //
    // Loads the dataset from the specified input-stream object, with
    // respect to the specified configuration. The configuration supplies
    // the mapping from raw label keys to `sample::y` target codes.
    //
    // If loading fails, `okay` remains false and `operator bool()` returns
    // false. (As with `configuration`, whether the constructor also throws
    // is decided by the implementation in dataset.cpp; this header
    // documents `okay` / `operator bool()` as the primary signal.)
    //
    // NOTE: `_is` is taken BY VALUE, not by reference. This is unusual for
    // stream parameters (they are typically non-copyable in practice and
    // passed by reference so the caller's stream position is visible).
    // It compiles only if `std::istream` is copy-constructible on your
    // standard library, which it generally is NOT. See the note below.
    dataset(std::istream _is, const config& _config);

    // Default construction is deleted: a dataset must always come from a
    // successfully loaded stream, so an "empty" instance is never a legal
    // state to have.
    dataset() = delete;
};

} // namespace PhantomX

#endif // DATASET_HPP