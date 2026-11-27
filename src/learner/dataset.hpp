// ============================================================================
//  dataset.hpp
// ============================================================================
//
//  Declares `dataset`, the in-memory container for training samples, their
//  certainty values and their targets.
//
//  Role of this type:
//
//    - `dataset` holds a flat vector of `sample`s. Each sample carries an
//      certainty ("u"), the attribute indices that describe the sample
//      ("x") and the packed target values ("y").
//
//    - It is loaded once from a stream against a `config`, which tells it how
//      label keys map to target codes. Unknown label keys are rejected during
//      validation. The loaded data is used as read-only input for learners.
//
//    - The class stores attribute *indices*, not attribute values: the
//      actual feature values live elsewhere and are looked up by index.
//
//  Sample representation:
//
//    Each `sample` has three parts:
//
//      - `certainty`: how uncertain this sample is, a `double` in the
//        half-open interval (0, 1]. Zero is excluded; one is allowed.
//      - `x`: the attribute indices that describe this sample.
//      - `y`: a fixed-size array of up to eight 16-bit target codes, one
//        per target goal (see `config::id()`). A slot left at its sentinel
//        value `0xffff` means "no target for this goal".
//
//  Validity:
//
//    A dataset is valid iff it is non-empty. The loader rejects an empty
//    input and empties the dataset whenever loading fails, so "non-empty"
//    and "loaded successfully" are the same thing.
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
// The in-memory collection of training samples plus their uncertainties and
// targets. Loading is delegated to the constructor; everything afterwards is
// read-only access.
// ----------------------------------------------------------------------------
struct dataset {

    // ------------------------------------------------------------------------
    // sample
    //
    // A single training example: its certainty, the attribute indices that
    // describe it, and its (up to eight) packed target codes.
    // ------------------------------------------------------------------------
    struct sample {

        // The certainty of this sample, in the half-open interval (0, 1].
        //
        // The loader guarantees the range for every sample it produces.
        // The default of 1.0 (maximum certainty) only matters for a
        // `sample` constructed by hand.
        double certainty = 1.0;

        // The attribute indices describing this sample.
        indexes x;

        // The target codes, one per goal. A slot left at the sentinel
        // value `0xffff` means "no target for this goal".
        //
        // The array is fixed at eight slots, so at most eight goals are
        // supported per sample (the configuration is responsible for
        // keeping its goal count within that limit). Each code is a
        // `config::id()` result, a 16-bit index into the configuration's
        // sorted "y", with `0xffff` reserved as the "absent" sentinel.
        // Slots beyond the configuration's goal count therefore keep the
        // sentinel value.
        uint16_t y[8] = {
            0xffff, 0xffff, 0xffff, 0xffff,
            0xffff, 0xffff, 0xffff, 0xffff
        };
    };

    // ------------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------------

    // The samples that make up the dataset. Empty if (and only if) the
    // dataset is not valid.
    std::vector<sample> samples;

public:

    // ------------------------------------------------------------------------
    // Validity
    // ------------------------------------------------------------------------

    // Boolean type-cast operator. Returns true iff the dataset is not
    // empty. Because the constructor empties the dataset whenever loading
    // fails, this is equivalent to "the dataset was loaded successfully".
    operator bool() const;

    // ------------------------------------------------------------------------
    // Access
    // ------------------------------------------------------------------------

    // Index operator. Returns a reference to the sample at the specified
    // index. The index is bounds-checked: an out-of-range index throws
    // `std::out_of_range`.
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
    // the number of labels every entry must carry and the mapping from raw
    // label keys to `sample::y` target codes.
    //
    // The stream is taken by reference, so the caller's stream position
    // reflects what was consumed (the whole stream is read).
    //
    // ValidationError is thrown internally for invalid input, caught by the
    // constructor, and reported on `std::cerr`. If loading fails, whether at
    // the very start or halfway through, the dataset is left empty, so
    // `operator bool()` returns false. No exception escapes the constructor.
    dataset(std::istream& _is, const config& _config);

    // Default construction is deleted: a dataset must always come from a
    // stream, so an unloaded instance is never a legal state to have.
    dataset() = delete;
};

} // namespace PhantomX

#endif // DATASET_HPP
