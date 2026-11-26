// ============================================================================
//  configuration.cpp
// ============================================================================
//
//  Implements `configuration` (aliased as `config`) and the internal JSON
//  loading & validation pipeline used to construct it from a stream[span_2](start_span)[span_2](end_span)[span_3](start_span)[span_3](end_span).
//
//  Loading pipeline overview:
//
//    1. Read the entire input stream into a `std::string` and check `_is.bad()`.
//    2. Parse JSON via `nlohmann::json::parse`.
//    3. Validate the root object and presence of `"x"`, `"y"`, and `"args"`.
//    4. Validate `"x"` and `"y"` into hash-sorted `config::group` vectors,
//       enforcing non-empty arrays, `[0, INT_MAX]` integers, no duplicate
//       indices within a group, and no hash collisions within a section.
//    5. Enforce mutual exclusivity between `"x"` and `"y"` keys.
//    6. Validate `"args"` into `config::args_space` (indices into sorted `y`),
//       requiring non-empty inner arrays of strings that exist in `"y"` with no
//       duplicates within a single inner array.
//    7. Mark `okay = true` only after all steps succeed; otherwise log the
//       error to `std::cerr` and leave `okay` as `false`[span_4](start_span)[span_4](end_span)[span_5](start_span)[span_5](end_span).
//
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <configuration.hpp>

namespace PhantomX {

using json = nlohmann::json;

namespace {

// ----------------------------------------------------------------------------
// Integer validation
// ----------------------------------------------------------------------------

// Validates that `value` is a JSON integer in the range [0, INT_MAX].
//
// Explicitly rejects booleans, null, strings, floating-point values (even
// whole-valued floats like `3.0`), negative integers, and values exceeding
// `INT_MAX`. `where` provides human-readable location context in error
// messages, and `int_max` caches `std::numeric_limits<int>::max()` as a
// 64-bit unsigned value for clean range comparisons.
int validate_int_value(const json& value, const std::string& where) {
    constexpr auto int_max = static_cast<std::uint64_t>(std::numeric_limits<int>::max());

    // nlohmann::json distinguishes booleans from numbers, but checking
    // explicitly makes the rejection intent self-documenting.
    if (value.is_boolean()) {
        throw ValidationError(where + " must be a non-negative integer, got a boolean.");
    }
    if (value.is_null()) {
        throw ValidationError(where + " must be a non-negative integer, got null.");
    }
    if (value.is_string()) {
        throw ValidationError(where + " must be a non-negative integer, got a string.");
    }
    if (value.is_number_float()) {
        throw ValidationError(where + " must be an integer, got a floating-point number.");
    }

    if (value.is_number_unsigned()) {
        const std::uint64_t v = value.get<std::uint64_t>();
        if (v > int_max) {
            throw ValidationError(
                where + " exceeds the maximum value of int (" + std::to_string(int_max) + ").");
        }
        return static_cast<int>(v);
    }

    if (value.is_number_integer()) {
        // Signed JSON integers may be negative; reject anything below 0 or
        // above `INT_MAX`.
        const std::int64_t v = value.get<std::int64_t>();
        if (v < 0) {
            throw ValidationError(
                where + " must be non-negative, got " + std::to_string(v) + ".");
        }
        if (static_cast<std::uint64_t>(v) > int_max) {
            throw ValidationError(
                where + " exceeds the maximum value of int (" + std::to_string(int_max) + ").");
        }
        return static_cast<int>(v);
    }

    throw ValidationError(
        where + " must be a non-negative integer, got " + std::string(value.type_name()) + ".");
}

// ----------------------------------------------------------------------------
// Root validation
// ----------------------------------------------------------------------------

// Verifies that `root` is a JSON object containing the required top-level keys
// `"x"`, `"y"`, and `"args"`. Extra root-level keys are permitted and ignored
// later.
void validate_root(const json& root) {
    if (!root.is_object()) {
        throw ValidationError(
            std::string("Root must be a JSON object, got ") + root.type_name() + ".");
    }
    for (const char* key : {"x", "y", "args"}) {
        if (!root.contains(key)) {
            throw ValidationError(
                std::string("Root object is missing required key \"") + key + "\".");
        }
    }
}

// ----------------------------------------------------------------------------
// "x" / "y" validation
// ----------------------------------------------------------------------------

// Validates a feature-grouping section (`"x"` or `"y"`) and converts it into a
// hash-sorted `config::groups` vector.
//
// Enforces:
//   - `obj` is a non-empty JSON object.
//   - Each key maps to a non-empty JSON array of valid `[0, INT_MAX]` ints.
//   - No duplicate integers appear within a single group's array.
//   - No `std::hash<std::string>` collisions occur among keys in `obj`.
//
// Key strings are hashed and immediately discarded; each group's index list
// is sorted ascending, and the resulting groups are sorted by `hash`.
config::groups validate_integer_map(const json& obj, const std::string& name) {
    config::groups groups;

    if (!obj.is_object()) {
        throw ValidationError(
            "\"" + name + "\" must be a JSON object, got " + obj.type_name() + ".");
    }
    if (obj.empty()) {
        throw ValidationError(
            "\"" + name + "\" must contain at least one key-value pair.");
    }

    groups.reserve(obj.size());

    // Tracks hash -> original key solely to report meaningful collision errors
    // before key strings are discarded.
    std::map<std::size_t, std::string> hash_owner;

    for (const auto& item : obj.items()) {
        const std::string& key = item.key();
        const json& arr = item.value();
        const std::string path = "\"" + name + "\"[\"" + key + "\"]";

        if (!arr.is_array()) {
            throw ValidationError(
                "Value of " + path + " must be an array, got " + arr.type_name() + ".");
        }
        if (arr.empty()) {
            throw ValidationError(
                "Array " + path + " must contain at least one element.");
        }

        std::vector<int> values;
        values.reserve(arr.size());
        for (std::size_t i = 0; i < arr.size(); ++i) {
            values.push_back(validate_int_value(
                arr[i], "Element " + path + "[" + std::to_string(i) + "]"));
        }

        // Sorting ascending places any duplicate values next to each other so
        // `std::adjacent_find` can detect repeats in O(n log n) time.
        std::sort(values.begin(), values.end());
        const auto dup = std::adjacent_find(values.begin(), values.end());
        if (dup != values.end()) {
            throw ValidationError(
                "Array " + path + " contains the value " + std::to_string(*dup) +
                " more than once; integers must not repeat within an array.");
        }

        const std::size_t hash = std::hash<std::string>{}(key);
        const auto inserted = hash_owner.emplace(hash, key);
        if (!inserted.second) {
            throw ValidationError(
                "Hash collision in \"" + name + "\": keys \"" + inserted.first->second +
                "\" and \"" + key + "\" have the same std::hash value.");
        }

        groups.push_back(config::group{std::move(values), hash});
    }

    // Sort groups by hash so `configuration::id()` can perform binary search.
    std::sort(groups.begin(), groups.end());
    return groups;
}

// ----------------------------------------------------------------------------
// Key separation
// ----------------------------------------------------------------------------

// Ensures `"x"` and `"y"` do not share any keys. Assumes both `x` and `y` have
// already been verified to be JSON objects.
void validate_key_separation(const json& x, const json& y) {
    for (const auto& item : x.items()) {
        if (y.contains(item.key())) {
            throw ValidationError(
                "Key \"" + item.key() +
                "\" appears in both \"x\" and \"y\"; keys must be mutually exclusive.");
        }
    }
}

// ----------------------------------------------------------------------------
// "args" validation
// ----------------------------------------------------------------------------

// Finds the index of `key` inside `sorted_y` (which is sorted by `group::hash`).
// Assumes `key` exists in `"y"` and hashes in `sorted_y` are unique.
int find_group_index(const config::groups& sorted_y, const std::string& key) {
    const std::size_t hash = std::hash<std::string>{}(key);
    const auto it = std::lower_bound(
        sorted_y.begin(), sorted_y.end(), hash,
        [](const config::group& g, std::size_t h) { return g.hash < h; });
    return static_cast<int>(it - sorted_y.begin());
}

// Validates `"args"` and translates each inner array of `"y"` key strings into
// indices into the already-sorted `y_groups` vector.
//
// Enforces:
//   - `args` is a non-empty JSON array.
//   - Each element of `args` is a non-empty JSON array of strings.
//   - Every string is a valid key in `"y"` (giving a specific diagnostic if it
//     belongs to `"x"` instead).
//   - No duplicate key strings appear within the same inner array (duplicates
//     across *different* inner arrays are allowed).
config::args_space validate_args(
    const json& args,
    const json& x,
    const json& y,
    const config::groups& y_groups) {
    config::args_space result;

    if (!args.is_array()) {
        throw ValidationError(
            std::string("\"args\" must be a JSON array, got ") + args.type_name() + ".");
    }
    if (args.empty()) {
        throw ValidationError("\"args\" must contain at least one inner array.");
    }

    result.reserve(args.size());

    for (std::size_t i = 0; i < args.size(); ++i) {
        const json& inner = args[i];
        const std::string inner_path = "\"args\"[" + std::to_string(i) + "]";

        if (!inner.is_array()) {
            throw ValidationError(
                "Element " + inner_path + " must be an array, got " + inner.type_name() + ".");
        }
        if (inner.empty()) {
            throw ValidationError(
                "Array " + inner_path + " must contain at least one element.");
        }

        // Duplicate checks are scoped strictly to the current inner array.
        std::set<std::string> seen;
        std::vector<int> indices;
        indices.reserve(inner.size());

        for (std::size_t j = 0; j < inner.size(); ++j) {
            const json& elem = inner[j];
            const std::string elem_path = inner_path + "[" + std::to_string(j) + "]";

            if (!elem.is_string()) {
                throw ValidationError(
                    "Element " + elem_path + " must be a string, got " + elem.type_name() + ".");
            }

            const std::string& s = elem.get_ref<const std::string&>();

            if (!y.contains(s)) {
                if (x.contains(s)) {
                    throw ValidationError(
                        "Element " + elem_path + " (\"" + s +
                        "\") is a key of \"x\", but only keys of \"y\" are allowed.");
                }
                throw ValidationError(
                    "Element " + elem_path + " (\"" + s +
                    "\") does not match any key in \"y\".");
            }

            if (!seen.insert(s).second) {
                throw ValidationError(
                    "Element " + elem_path + " (\"" + s +
                    "\") is a duplicate within " + inner_path + ".");
            }

            indices.push_back(find_group_index(y_groups, s));
        }

        result.push_back(std::move(indices));
    }
    
    if (result.size() > 8) {
        throw ValidationError(
            "Arguments must be at most 8; whereas there are "
            + std::to_string(result.size()) + ".");
    }

    return result;
}

} // namespace

// ----------------------------------------------------------------------------
// configuration member functions
// ----------------------------------------------------------------------------

// Gets the 16-bit ID of the key with the specified hash, or std::nullopt
// if no key in `y` has that hash.
//
// The ID is the group's position in the hash-sorted `y` vector. It is
// stable for a given configuration but NOT across configurations
// (inserting a key can renumber existing keys).
//
// Returns std::nullopt if the hash is not present.
std::optional<std::uint16_t> id(std::size_t _y_key) const {
    std::size_t lo = 0;
    std::size_t hi = y.size();

    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (y[mid].hash == _y_key) {
            return static_cast<std::uint16_t>(mid);
        }
        if (y[mid].hash < _y_key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    return std::nullopt;
}

// Returns `true` if the configuration finished loading and validating without
// error[span_8](start_span)[span_8](end_span)[span_9](start_span)[span_9](end_span).
configuration::operator bool() const {
    return okay;
}

// Reads and validates a JSON configuration from `_is`[span_10](start_span)[span_10](end_span)[span_11](start_span)[span_11](end_span).
//
// Expected schema:
//
//   {
//     "x":    { "<key>": [non-negative integers, at least one], ... },
//     "y":    { "<key>": [non-negative integers, at least one], ... },
//     "args": [ ["<key in y>", ...], ... ]
//   }
//
// On any I/O, parse, or validation error, a diagnostic is written to
// `std::cerr` and `okay` remains `false` (exceptions are caught internally
// rather than propagated out of the constructor)[span_12](start_span)[span_12](end_span)[span_13](start_span)[span_13](end_span).
configuration::configuration(std::istream& _is) {
    json root;

    try {
        const std::string content(
            (std::istreambuf_iterator<char>(_is)),
            std::istreambuf_iterator<char>());

        if (_is.bad()) {
            throw std::runtime_error("I/O error while reading file.");
        }

        root = json::parse(content);
        validate_root(root);

        const json& x_json = root.at("x");
        const json& y_json = root.at("y");
        const json& args_json = root.at("args");

        // Validate and populate `x` and `y` before `validate_args` maps `"y"`
        // keys into indices into the hash-sorted `y` vector.
        x = validate_integer_map(x_json, "x");
        y = validate_integer_map(y_json, "y");

        validate_key_separation(x_json, y_json);

        args = validate_args(args_json, x_json, y_json, y);

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
