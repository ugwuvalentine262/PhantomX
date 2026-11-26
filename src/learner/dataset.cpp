// ============================================================================
//  dataset.cpp
// ============================================================================
//
//  Implements `PhantomX::dataset` (declared in dataset.hpp): the loader that
//  turns a JSON document into a vector of validated training samples.
//
//  Expected input format:
//
//    The stream must contain a single, non-empty JSON array. Every element
//    of that array is itself a two-element array, "an entry":
//
//        [
//          [ ["label_a", "label_b"], [3, 17, 42] ],   // entry 0
//          [ ["label_a", "label_c"], [0, 5]      ],   // entry 1
//          ...
//        ]
//
//    where, for every entry:
//
//      - element 0 ("y") is an array of exactly `k` strings, with `k` being
//        `config::args.size()`. Each string is a label key. It is hashed and
//        mapped through `config::id()` to a 16-bit target code that is
//        stored in `sample::y`.
//
//      - element 1 ("x") is an array of non-negative integers that fit in an
//        `int`: the attribute indices describing the sample. The values are
//        sorted after loading, and duplicates are rejected, so every
//        `sample::x` is sorted, non-negative and non-repeating (the
//        invariant that `indexes` relies on).
//
//  Error handling strategy:
//
//    The constructor NEVER lets an exception escape. Every failure (bad
//    JSON, failed validation, I/O trouble, ...) is reported on `std::cerr`
//    and leaves `dataset::okay` at `false`. Callers must test the dataset
//    (via `operator bool()`) before using it.
//
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <dataset.hpp>

using json = nlohmann::json;

namespace PhantomX {

// ----------------------------------------------------------------------------
// read_non_negative_int  (file-local helper)
//
// Validates that a JSON value is a non-negative integer that fits in an `int`
// and, if so, stores it in `out`.
//
// The checks are deliberately strict, and are performed from the most
// specific rejection to the most general one so that the error message is as
// helpful as possible:
//
//   1. Booleans are rejected. (nlohmann::json keeps booleans as their own
//      type, but being explicit gives a clearer message than the generic
//      "not an integer" one.)
//   2. Floating-point numbers are rejected, even when they hold a whole
//      value such as `3.0`.
//   3. Anything else that is not an integer is rejected.
//   4. Negative integers are rejected.
//   5. Integers larger than `INT_MAX` are rejected.
//
// Parameters:
//   value - the JSON value to inspect.
//   out   - receives the converted integer on success. It is left untouched
//           on failure.
//
// Returns:
//   An empty string on success, otherwise a short human-readable description
//   of the problem. The caller is responsible for prefixing it with the
//   location (entry / index) of the offending value.
//
// Linkage:
//   `static`, so the helper is private to this translation unit.
// ----------------------------------------------------------------------------
static std::string read_non_negative_int(const json& value, int& out) {

    // --- Step 1: reject anything that is not a plain integer --------------

    if (value.is_boolean()) {
        return "boolean is not an integer";
    }
    if (value.is_number_float()) {
        return "floating-point number is not allowed";
    }
    // `is_number_integer()` is true for BOTH signed and unsigned JSON
    // integers, so it is the right catch-all for "some kind of integer".
    if (!value.is_number_integer()) {
        return "value is not an integer";
    }

    // --- Step 2: extract the magnitude as an unsigned 64-bit value ---------
    //
    // nlohmann::json stores an integer as either int64_t or uint64_t,
    // depending on how the literal was parsed (anything above INT64_MAX is
    // unsigned). Widening everything to uint64_t lets the single range check
    // below handle both representations, while the sign is checked first.

    std::uint64_t magnitude = 0;

    if (value.is_number_unsigned()) {
        // An unsigned value can never be negative; take it as is.
        magnitude = value.get<std::uint64_t>();
    } else {
        // A signed value must be checked for a negative sign before it is
        // converted, because casting a negative int64_t to uint64_t would
        // silently wrap around to a huge positive number.
        const std::int64_t signed_value = value.get<std::int64_t>();
        if (signed_value < 0) {
            return "negative integers are not allowed";
        }
        magnitude = static_cast<std::uint64_t>(signed_value);
    }

    // --- Step 3: make sure the value fits in an `int` ----------------------

    if (magnitude >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        return "integer is outside the supported int range";
    }

    out = static_cast<int>(magnitude);
    return std::string();
}

// ----------------------------------------------------------------------------
// dataset::operator bool
//
// Boolean type-cast operator. Returns true iff the dataset was loaded
// successfully by the constructor (that is, iff `okay` is set).
// ----------------------------------------------------------------------------
dataset::operator bool() const {
    return dataset::okay;
}

// ----------------------------------------------------------------------------
// dataset::operator[]
//
// Index operator. Returns a reference to the sample at the specified index.
//
// NOTE: this uses `std::vector::at()`, so an out-of-range index throws
// `std::out_of_range` instead of being undefined behaviour. (dataset.hpp
// currently documents the access as unchecked; the two should be brought in
// line.)
// ----------------------------------------------------------------------------
const dataset::sample& dataset::operator[](std::size_t _index) const {
    return dataset::samples.at(_index);
}

// ----------------------------------------------------------------------------
// dataset::size
//
// Gets the number of samples in the dataset.
// ----------------------------------------------------------------------------
std::size_t dataset::size() const {
    return dataset::samples.size();
}

// ----------------------------------------------------------------------------
// dataset::dataset  (constructor)
//
// Loads the dataset from the specified input-stream object, with respect to
// the specified configuration.
//
// Overview of the algorithm:
//
//   1. Read the whole stream into a string.
//   2. Check the configuration, and the stream state.
//   3. Parse the string as JSON.
//   4. Validate the document's overall shape (non-empty array).
//   5. For every entry, validate its shape, convert the labels ("y") into
//      target codes, convert the attribute indices ("x"), and append the
//      resulting `sample`.
//   6. Only when every entry was accepted, set `okay` to true.
//
// All failures are reported on `std::cerr` and leave `okay == false`; the
// constructor never throws. Be aware that `samples` may be left partially
// filled when loading fails midway, so it must not be used unless `okay` is
// true.
//
// Parameters:
//   _is     - the stream to read the JSON document from.
//   _config - the configuration that supplies `args` (the number of labels
//             every entry must carry) and `id()` (the label-to-code map).
//
// NOTE: `_is` is taken BY VALUE, exactly as declared in dataset.hpp.
// `std::istream` is not copyable, so this signature will not compile on a
// conforming standard library; it should become `std::istream&` in both the
// header and here.
// ----------------------------------------------------------------------------
dataset::dataset(std::istream _is, const config& _config) {

    // The number of labels ("y" strings) every entry must contain.
    const std::size_t k = _config.args.size();

    // The parsed document. Declared outside the `try` for readability only.
    json root;

    try {
        // Slurp the entire stream into memory. nlohmann::json parses from a
        // complete string/iterator range, so the stream is read in one go
        // (character by character through the stream buffer, which is fast).
        const std::string content((std::istreambuf_iterator<char>(_is)),
                                   std::istreambuf_iterator<char>());

        // --------------------------------------------------------------------
        // Pre-flight checks
        // --------------------------------------------------------------------

        // An undefined configuration cannot map labels to codes, so there is
        // no point in going any further.
        if (!_config) {
            throw ValidationError(
                "Error: specified configuration is not defined.");
        }

        // Make sure reading the stream did not hit an unrecoverable I/O
        // error (`badbit`). Note that `failbit` / `eofbit` are expected after
        // reading to the end of the stream, so only `bad()` is meaningful
        // here.
        if (_is.bad()) {
            throw std::runtime_error("failed while reading dataset from istream.");
        }

        // Parse the text. Throws `json::parse_error` on malformed JSON,
        // which is handled by the catch blocks at the bottom.
        root = json::parse(content);

        // --------------------------------------------------------------------
        // Root validation: a non-empty array of entries
        // --------------------------------------------------------------------

        if (!root.is_array()) {
            throw ValidationError("Error: root JSON value must be an array.");
        }
        if (root.empty()) {
            throw ValidationError("Error: root array must not be empty.");
        }

        // One sample is produced per entry, so the final size is known and
        // the vector can be allocated up front (avoids repeated growth).
        samples.reserve(root.size());

        // --------------------------------------------------------------------
        // Entry validation and conversion
        // --------------------------------------------------------------------

        for (std::size_t i = 0; i < root.size(); ++i) {

            const json& entry = root[i];

            // Prefix used by every error message about this entry, so that
            // the user can find the offending spot in the input file.
            const std::string where = "entry " + std::to_string(i);

            // --- Shape of the entry: [ [labels...], [attributes...] ] ------

            if (!entry.is_array()) {
                throw ValidationError(where + ": entry must be an array.");
            }
            if (entry.size() != 2) {
                throw ValidationError(
                    where + ": entry must contain exactly 2 elements, found " +
                    std::to_string(entry.size()) + ".");
            }
            if (!entry[0].is_array()) {
                throw ValidationError(
                    where + ", element 0: must be an array of strings.");
            }
            if (!entry[1].is_array()) {
                throw ValidationError(
                    where + ", element 1: must be an array of integers.");
            }

            // Short names matching the notation used by `sample`:
            //   y - the label keys (targets),
            //   x - the attribute indices (features).
            const json& y = entry[0];
            const json& x = entry[1];

            // --- Shape of the label array ----------------------------------

            if (y.empty()) {
                throw ValidationError(
                    where + ", element 0: string array must not be empty.");
            }
            // Every entry must carry exactly one label per configured goal.
            //
            // CAUTION: `sample::y` is a fixed array of eight slots and the
            // loop below indexes it with `j < k`. Nothing here (or in this
            // function) guarantees `k <= 8`, so a configuration with more
            // than eight goals would write out of bounds. A check on `k`
            // before the loop would make this safe.
            if (y.size() != k) {
                throw ValidationError(
                    where + ", element 0: string array must contain exactly " +
                    std::to_string(k) + " strings, found " +
                    std::to_string(y.size()) + ".");
            }

            // The sample being built. Its name intentionally shadows the
            // type name `dataset::sample` from this point on (legal, but
            // easy to trip over: `sample.y` below is the variable).
            //
            // `sample::y` starts out filled with the 0xffff "no target"
            // sentinel (see dataset.hpp), so any slot not assigned below
            // keeps that meaning.
            sample sample;

            // --- Labels: strings -> 16-bit target codes --------------------
            //
            // Each label key is hashed with `std::hash<std::string>` and the
            // hash is handed to `config::id()`, which returns the code for
            // that key (or 0xffff when the key is unknown to the
            // configuration). An unknown label is therefore stored as the
            // "no target" sentinel rather than being reported as an error.
            //
            // Also note that `std::hash` values are implementation-defined,
            // so they are only meaningful when `config::id()` hashes keys
            // with the same function (same standard library and build).

            for (std::size_t j = 0; j < y.size(); ++j) {
                if (!y[j].is_string()) {
                    throw ValidationError(
                        where + ", string index " + std::to_string(j) +
                        ": element must be a string.");
                }
                sample.y[j] = *_config.id(
                    std::hash<std::string>{}(y[j].get<std::string>()));
            }

            // --- Attributes: JSON integers -> sorted `indexes` -------------

            sample.x.reserve(x.size());

            for (std::size_t j = 0; j < x.size(); ++j) {
                int value = 0;

                // Strict integer validation; see read_non_negative_int().
                const std::string problem = read_non_negative_int(x[j], value);
                if (!problem.empty()) {
                    throw ValidationError(
                        where + ", integer index " + std::to_string(j) + ": " +
                        problem + ".");
                }
                sample.x.push_back(value);
            }

            // `indexes` is required to be sorted (and non-repeating), so
            // sort now regardless of the order used in the input file.
            std::sort(sample.x.begin(), sample.x.end());

            // After sorting, equal values are adjacent, so one linear pass
            // comparing neighbours is enough to detect duplicates.
            // (`x.size()` equals `sample.x.size()` here.)
            for (std::size_t j = 1; j < x.size(); ++j) {
                if (sample.x[j - 1] == sample.x[j]) {
                    throw ValidationError(where + ", duplicate values.");
                }
            }

            // The sample is fully validated; move it into the dataset
            // instead of copying its vector.
            samples.push_back(std::move(sample));
        }

        // Reached only if every entry was accepted. This is the single place
        // where the dataset is declared valid.
        okay = true;

    // ------------------------------------------------------------------------
    // Error reporting
    //
    // Handlers are ordered from the most specific exception type to the most
    // general one, because the first matching handler wins:
    //
    //   json::parse_error  -> the text is not valid JSON
    //   ValidationError    -> valid JSON, but not the expected structure
    //   json::exception    -> any other nlohmann::json failure
    //   std::exception     -> anything else (e.g. the I/O error above)
    //
    // In all cases `okay` stays false.
    // ------------------------------------------------------------------------
    } catch (const json::parse_error& e) {
        std::cerr << "Error: malformed JSON: " << e.what() << '\n';
    } catch (const ValidationError& e) {
        std::cerr << "Error: invalid input: " << e.what() << '\n';
    } catch (const json::exception& e) {
        std::cerr << "Error: JSON processing failure: " << e.what() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
}

} // namespace PhantomX
