// ============================================================================
//  configuration.cpp
// ============================================================================
//
//  Implements `configuration` (aliased as `config`) and the internal JSON
//  loading & validation pipeline used to construct it from a stream.
//
//  Loading pipeline overview:
//
//    1. Read the entire input stream into a `std::string` and check `_is.bad()`.
//    2. Parse JSON via `nlohmann::json::parse`.
//    3. Validate the root object and presence of `"x"`, `"y"`, and `"args"`.
//    4. Validate `"x"` and `"y"` into hash-sorted `config::group` vectors,
//       enforcing non-empty objects, non-empty arrays, `[0, INT_MAX]`
//       integers, no duplicate indices within a group, and no hash collisions
//       within a section. `"y"` is additionally limited to
//       `config::max_y_groups` (65536) keys so that every position in the
//       sorted `y` vector fits in the 16-bit ID returned by `id()`.
//    5. Enforce mutual exclusivity between `"x"` and `"y"` keys.
//    6. Validate `"args"` into `config::args_space` (indices into sorted `y`),
//       requiring at most `config::max_args` (8) non-empty inner arrays of
//       strings that exist in `"y"` with no duplicates within a single inner
//       array. The order written in the JSON is preserved.
//    7. If every step succeeds, `x`, `y` and `args` are all non-empty, which is
//       exactly what `operator bool()` checks. There is no separate flag.
//
//  Failure handling:
//
//    The constructor never throws. Every exception, including non-standard
//    ones, is caught inside the constructor. Each failure writes one
//    diagnostic line to `std::cerr`, and then `x`, `y` and `args` are all
//    cleared, so a failed configuration is completely empty and
//    `operator bool()` returns false.
//
//  Known limitation:
//
//    A key repeated within one JSON object is not detected: nlohmann::json
//    keeps only the last occurrence, so the earlier ones are dropped before
//    validation sees them.
//
//  How this file is organised:
//
//    - An anonymous namespace holds every helper (hashing, integer checks,
//      per-section validators). Being in an anonymous namespace gives them
//      internal linkage: they are invisible to other translation units and
//      cannot clash with names elsewhere in PhantomX.
//    - All helpers report problems by THROWING `ValidationError` (or letting
//      nlohmann::json throw). They never log and never return error codes.
//      The single place that logs and recovers is the constructor at the
//      bottom of the file, which keeps the error policy in one spot.
//    - The member functions of `configuration` come after the helpers.
//
// ============================================================================

// ---- Standard library ------------------------------------------------------
#include <algorithm>     // std::sort, std::adjacent_find, std::lower_bound
#include <cstdint>       // std::uint16_t / std::uint64_t / std::int64_t
#include <iostream>      // std::cerr (the constructor's diagnostics)
#include <iterator>      // std::istreambuf_iterator (reading the whole stream)
#include <limits>        // std::numeric_limits (INT_MAX, "no limit" default)
#include <map>           // std::map (hash -> key, for collision messages)
#include <set>           // std::set (duplicate detection inside "args")
#include <string>        // std::string, std::to_string
#include <string_view>   // std::string_view (parameter type of fnv1a)
#include <utility>       // std::move
#include <vector>        // std::vector

// ---- Third party / project -------------------------------------------------
#include <nlohmann/json.hpp>
#include <configuration.hpp>

namespace PhantomX {

// Short alias so the code below reads `json` instead of `nlohmann::json`.
using json = nlohmann::json;

namespace {

// ----------------------------------------------------------------------------
// Hashing
// ----------------------------------------------------------------------------

// Computes the 64-bit FNV-1a hash of `key`.
//
// Starts from the offset basis, then for every byte of the key XORs the byte
// into the hash and multiplies by the FNV prime (modulo 2^64, which unsigned
// arithmetic gives for free). Bytes are read as `unsigned char` so the result
// does not depend on whether `char` is signed on the platform.
//
// This is the hash stored in `config::group::hash` and used by
// `configuration::id()`. It is deterministic across platforms and runs.
// Distinct keys can in principle collide; the constructor rejects such
// collisions within "x" or "y". Never throws and does not allocate.
std::uint64_t fnv1a(std::string_view key) noexcept {
    // The two constants below are the official 64-bit FNV parameters.
    // Changing either one changes every hash (and therefore every ID), so
    // they must never be edited casually.
    constexpr std::uint64_t offset_basis = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;

    // Running hash value, seeded with the offset basis. An empty key
    // therefore hashes to the offset basis itself.
    std::uint64_t h = offset_basis;

    for (const char c : key) {
        // Step 1 of FNV-1a: XOR the byte into the low bits of the hash.
        // The double cast is deliberate: first to `unsigned char` so that a
        // negative `char` (bytes >= 0x80 where char is signed) becomes
        // 0..255 instead of sign-extending, then to 64 bits for the XOR.
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));

        // Step 2 of FNV-1a: multiply by the prime. Unsigned overflow is
        // well-defined (wraps modulo 2^64), which is exactly what the
        // algorithm requires.
        h *= prime;
    }
    return h;
}

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
    // INT_MAX widened to 64-bit unsigned. Comparing in this wider type avoids
    // signed/unsigned comparison warnings and any overflow when the JSON
    // number does not fit in an `int`.
    constexpr auto int_max = static_cast<std::uint64_t>(std::numeric_limits<int>::max());

    // ---- Step 1: reject every non-numeric type with a specific message ----
    //
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

    // ---- Step 2: reject floating-point numbers ----
    //
    // Even a whole-valued float such as 3.0 is refused: the schema asks for
    // integers, and accepting "3.0" would hide typos in generated files.
    if (value.is_number_float()) {
        // Integers too large for 64 bits are also parsed as floating point by
        // nlohmann::json, so the message mentions both cases.
        throw ValidationError(
            where + " must be a non-negative integer, got a floating-point number"
            " (or an integer too large for 64 bits).");
    }

    // ---- Step 3: integers that nlohmann stored as unsigned ----
    //
    // nlohmann stores non-negative integers parsed from text as unsigned
    // 64-bit values, so this is the path every VALID input normally takes.
    // Only the upper bound needs checking here; it can't be negative.
    if (value.is_number_unsigned()) {
        const std::uint64_t v = value.get<std::uint64_t>();
        if (v > int_max) {
            throw ValidationError(
                where + " exceeds the maximum value of int (" + std::to_string(int_max) + ").");
        }
        return static_cast<int>(v);
    }

    // ---- Step 4: integers that nlohmann stored as signed ----
    //
    // Negative integers arrive here (and, depending on how the document was
    // built, possibly small positive ones too), so both bounds are checked.
    if (value.is_number_integer()) {
        // Signed JSON integers may be negative; reject anything below 0 or
        // above `INT_MAX`.
        const std::int64_t v = value.get<std::int64_t>();
        if (v < 0) {
            throw ValidationError(
                where + " must be non-negative, got " + std::to_string(v) + ".");
        }
        // `v` is now known to be >= 0, so converting to unsigned is safe and
        // lets us compare against `int_max` without a sign mismatch.
        if (static_cast<std::uint64_t>(v) > int_max) {
            throw ValidationError(
                where + " exceeds the maximum value of int (" + std::to_string(int_max) + ").");
        }
        return static_cast<int>(v);
    }

    // ---- Step 5: defensive fallback ----
    //
    // Reached only for JSON types that were not handled above (arrays,
    // objects, binary, ...). `type_name()` gives a readable name such as
    // "array" or "object" for the message.
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
    // The document must be `{ ... }`, not an array, string, number, etc.
    if (!root.is_object()) {
        throw ValidationError(
            std::string("Root must be a JSON object, got ") + root.type_name() + ".");
    }

    // Only presence is checked here. The type and contents of each section
    // are validated later by the section-specific validators, which can give
    // far more precise messages.
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
//   - `obj` has at most `max_groups` keys (checked before any other work on
//     the entries; the default means "no limit").
//   - Each key maps to a non-empty JSON array of valid `[0, INT_MAX]` ints.
//   - No duplicate integers appear within a single group's array.
//   - No FNV-1a hash collisions (see `fnv1a()`) occur among keys in
//     `obj`.
//
// Key strings are hashed and immediately discarded; each group's index list
// is sorted ascending, and the resulting groups are sorted by `hash`.
//
// Parameters:
//   obj         the JSON value of the section ("x" or "y").
//   name        the section's name, used only to build error messages.
//   max_groups  upper bound on the number of keys. The default (the largest
//               size_t) effectively means "unlimited"; the constructor passes
//               `max_y_groups` for "y" and nothing for "x".
config::groups validate_integer_map(
    const json& obj,
    const std::string& name,
    std::size_t max_groups = std::numeric_limits<std::size_t>::max()) {
    // The result, built up one group per JSON key.
    config::groups groups;

    // ---- Section-level checks (cheap, so done before touching entries) ----
    if (!obj.is_object()) {
        throw ValidationError(
            "\"" + name + "\" must be a JSON object, got " + obj.type_name() + ".");
    }
    if (obj.empty()) {
        throw ValidationError(
            "\"" + name + "\" must contain at least one key-value pair.");
    }
    // Size limit. Doing this before the loop means an oversized section is
    // rejected immediately, without hashing or sorting tens of thousands of
    // entries first.
    if (obj.size() > max_groups) {
        throw ValidationError(
            "\"" + name + "\" must contain at most " + std::to_string(max_groups) +
            " keys, got " + std::to_string(obj.size()) + ".");
    }

    // We know exactly how many groups we will produce, so allocate once.
    groups.reserve(obj.size());

    // Tracks hash -> original key solely to report meaningful collision errors
    // before key strings are discarded.
    std::map<std::uint64_t, std::string> hash_owner;

    // ---- Per-entry checks ----
    // `obj.items()` yields key/value pairs. nlohmann's default object type is
    // a std::map, so keys arrive in lexicographic order; the final order of
    // `groups` does NOT depend on this because it is re-sorted by hash below.
    for (const auto& item : obj.items()) {
        const std::string& key = item.key();
        const json& arr = item.value();

        // Human-readable location, e.g.  "x"["age"]  , reused in messages.
        const std::string path = "\"" + name + "\"[\"" + key + "\"]";

        // The value for every key must be a non-empty array.
        if (!arr.is_array()) {
            throw ValidationError(
                "Value of " + path + " must be an array, got " + arr.type_name() + ".");
        }
        if (arr.empty()) {
            throw ValidationError(
                "Array " + path + " must contain at least one element.");
        }

        // Validate each element and collect the results. `validate_int_value`
        // throws on the first bad element, with its exact position in `where`.
        std::vector<int> values;
        values.reserve(arr.size());
        for (std::size_t i = 0; i < arr.size(); ++i) {
            values.push_back(validate_int_value(
                arr[i], "Element " + path + "[" + std::to_string(i) + "]"));
        }

        // Sorting ascending places any duplicate values next to each other so
        // `std::adjacent_find` can detect repeats in O(n log n) time.
        // The sort also establishes the documented invariant that a group's
        // indices are ascending.
        std::sort(values.begin(), values.end());
        const auto dup = std::adjacent_find(values.begin(), values.end());
        if (dup != values.end()) {
            throw ValidationError(
                "Array " + path + " contains the value " + std::to_string(*dup) +
                " more than once; integers must not repeat within an array.");
        }

        // Hash the key, then try to register the hash. `emplace` fails (second
        // == false) if another key already produced the same hash, in which
        // case `first->second` is the key that got there first, so both names
        // can be reported.
        const std::uint64_t hash = fnv1a(key);
        const auto inserted = hash_owner.emplace(hash, key);
        if (!inserted.second) {
            throw ValidationError(
                "Hash collision in \"" + name + "\": keys \"" + inserted.first->second +
                "\" and \"" + key + "\" have the same FNV-1a hash.");
        }

        // Keep only the hash and the sorted indices. `values` is moved (not
        // copied) into the group, and the key string is dropped here.
        groups.push_back(config::group{std::move(values), hash});
    }

    // Sort groups by hash so `configuration::id()` can perform binary search.
    // This uses `group::operator<`, which compares by hash only.
    std::sort(groups.begin(), groups.end());
    return groups;
}

// ----------------------------------------------------------------------------
// Key separation
// ----------------------------------------------------------------------------

// Ensures `"x"` and `"y"` do not share any keys. Assumes both `x` and `y` have
// already been verified to be JSON objects.
void validate_key_separation(const json& x, const json& y) {
    // It is enough to walk one side and look each key up in the other: if any
    // key is common to both, this loop finds it.
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
//
// Because the key strings are not stored, the lookup is done on the key's
// hash. `std::lower_bound` returns the first group whose hash is NOT less than
// the target; given the assumptions above, that is exactly the matching group,
// so its distance from `begin()` is the key's position (its ID).
int find_group_index(const config::groups& sorted_y, const std::string& key) {
    const std::uint64_t hash = fnv1a(key);
    const auto it = std::lower_bound(
        sorted_y.begin(), sorted_y.end(), hash,
        // Comparator: "is this group's hash smaller than the target hash?"
        // lower_bound requires the element on the left, value on the right.
        [](const config::group& g, std::uint64_t h) { return g.hash < h; });
    // The position fits in an int: `sorted_y` has at most 65536 entries.
    return static_cast<int>(it - sorted_y.begin());
}

// Validates `"args"` and translates each inner array of `"y"` key strings into
// indices into the already-sorted `y_groups` vector.
//
// Enforces:
//   - `args` is a non-empty JSON array with at most `config::max_args` (8)
//     elements; this is checked first, before any element is examined.
//   - Each element of `args` is a non-empty JSON array of strings.
//   - Every string is a valid key in `"y"` (giving a specific diagnostic if it
//     belongs to `"x"` instead).
//   - No duplicate key strings appear within the same inner array (duplicates
//     across *different* inner arrays are allowed).
//
// Parameters:
//   args      the JSON value of the "args" section.
//   x, y      the raw JSON objects of "x" and "y". Only their KEYS are
//             consulted here, to test membership and to tell "wrong section"
//             apart from "unknown key" in error messages.
//   y_groups  the already-built, hash-sorted groups of "y"; used to translate
//             key strings into positions.
config::args_space validate_args(
    const json& args,
    const json& x,
    const json& y,
    const config::groups& y_groups) {
    // The result: one `indexes` (vector<int>) per inner array, in the same
    // order as written in the JSON. Order inside each inner array is also
    // preserved because it is meaningful ("ordered sequence of y keys").
    config::args_space result;

    // ---- Section-level checks ----
    if (!args.is_array()) {
        throw ValidationError(
            std::string("\"args\" must be a JSON array, got ") + args.type_name() + ".");
    }
    if (args.empty()) {
        throw ValidationError("\"args\" must contain at least one inner array.");
    }
    // Upper bound. Checked up front so a document with, say, thousands of
    // inner arrays is refused without examining any of them.
    if (args.size() > config::max_args) {
        throw ValidationError(
            "\"args\" must contain at most " + std::to_string(config::max_args) +
            " inner arrays, got " + std::to_string(args.size()) + ".");
    }

    result.reserve(args.size());

    // ---- Per inner array ----
    for (std::size_t i = 0; i < args.size(); ++i) {
        const json& inner = args[i];

        // Location used in messages, e.g.  "args"[2]  .
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
        // `seen` is created fresh for every inner array, which is exactly why
        // the same key MAY appear in different inner arrays.
        std::set<std::string> seen;
        std::vector<int> indices;
        indices.reserve(inner.size());

        // ---- Per key string ----
        for (std::size_t j = 0; j < inner.size(); ++j) {
            const json& elem = inner[j];

            // Location used in messages, e.g.  "args"[2][0]  .
            const std::string elem_path = inner_path + "[" + std::to_string(j) + "]";

            if (!elem.is_string()) {
                throw ValidationError(
                    "Element " + elem_path + " must be a string, got " + elem.type_name() + ".");
            }

            // Borrow the string stored inside the JSON value (no copy).
            // Safe because `elem` is known to be a string just above.
            const std::string& s = elem.get_ref<const std::string&>();

            // The key must belong to "y". If it doesn't, check "x" so the
            // user is told the more helpful "wrong section" message instead
            // of a generic "unknown key".
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

            // `insert(...).second` is false if the string was already in the
            // set, i.e. it appeared earlier in THIS inner array.
            if (!seen.insert(s).second) {
                throw ValidationError(
                    "Element " + elem_path + " (\"" + s +
                    "\") is a duplicate within " + inner_path + ".");
            }

            // Translate the key into its position in the hash-sorted `y`.
            // Safe: membership in "y" was verified just above, and hash
            // uniqueness within "y" was verified when `y_groups` was built.
            indices.push_back(find_group_index(y_groups, s));
        }

        // Move (not copy) the finished inner array into the result.
        result.push_back(std::move(indices));
    }

    return result;
}

} // namespace

// ----------------------------------------------------------------------------
// configuration member functions
// ----------------------------------------------------------------------------

// Gets the 16-bit ID of the key `_key` in `y`, or std::nullopt if `y` has no
// such key.
//
// `_key` is hashed with `fnv1a()`; the resulting hash (`_y_key` below) is then
// looked up by binary search in the hash-sorted `y` vector.
//
// The ID is the group's position in the hash-sorted `y` vector. It is
// stable for a given configuration but NOT across configurations
// (inserting a key can renumber existing keys). Because the constructor
// limits `y` to `max_y_groups` (65536) entries, the position always fits in
// 16 bits and the cast below never truncates.
//
// Returns std::nullopt if the key is not present (which includes the case of
// an invalid, empty configuration). Never throws: hashing and searching do
// not allocate.
std::optional<std::uint16_t> configuration::id(const std::string& _key) const noexcept {
    // Hash of the string. This is the value compared against `group::hash`.
    const std::uint64_t _y_key = fnv1a(_key);

    // Classic binary search over the half-open range [lo, hi). It relies on
    // `y` being sorted ascending by hash, which the constructor guarantees.
    // On an empty `y` (invalid configuration) the loop body never runs and
    // the function returns std::nullopt.
    std::size_t lo = 0;
    std::size_t hi = y.size();

    while (lo < hi) {
        // `lo + (hi - lo) / 2` instead of `(lo + hi) / 2` avoids any
        // possibility of overflow in the addition.
        const std::size_t mid = lo + (hi - lo) / 2;

        if (y[mid].hash == _y_key) {
            // Found: the position IS the ID. At most 65535 here, so the
            // narrowing to 16 bits is lossless (see the limit above).
            return static_cast<std::uint16_t>(mid);
        }
        if (y[mid].hash < _y_key) {
            // Target is larger: discard the lower half, including `mid`.
            lo = mid + 1;
        } else {
            // Target is smaller: discard the upper half, including `mid`.
            hi = mid;
        }
    }

    // Range exhausted without a match.
    return std::nullopt;
}

// Returns `true` iff `x`, `y` and `args` are all non-empty, and `false` if any
// of them is empty.
//
// A successful load always produces three non-empty members and a failed load
// clears all three, so this is equivalent to "the constructor succeeded".
// Never throws.
configuration::operator bool() const noexcept {
    return !x.empty() && !y.empty() && !args.empty();
}

// Reads and validates a JSON configuration from `_is`.
//
// Expected schema:
//
//   {
//     "x":    { "<key>": [non-negative integers, at least one], ... },
//     "y":    { "<key>": [non-negative integers, at least one], ... },
//     "args": [ ["<key in y>", ...], ... ]
//   }
//
// Limits: "y" may hold at most `max_y_groups` keys and "args" at most
// `max_args` inner arrays (see configuration.hpp).
//
// Error handling: this constructor never throws. On any I/O, parse, or
// validation error, or on any other exception (including non-standard ones),
// a diagnostic line is written to `std::cerr`, and then `x`, `y` and `args`
// are all cleared so that `operator bool()` returns false.
configuration::configuration(std::istream& _is) {
    // Holds the parsed document. Declared outside the `try` only so that it
    // lives for the whole constructor; it is not used after the `try` block.
    json root;

    try {
        // ---- Step 1: read the whole stream into memory ----
        //
        // Iterating with `istreambuf_iterator` copies raw characters straight
        // from the stream's buffer until end-of-stream. Unlike `operator>>` it
        // does not skip whitespace, so the text reaches the parser untouched.
        // The extra parentheses around the first argument avoid the "most
        // vexing parse" (without them this would declare a function).
        const std::string content(
            (std::istreambuf_iterator<char>(_is)),
            std::istreambuf_iterator<char>());

        // Reject a stream that is in a bad state. Note that reading through
        // the stream buffer directly does not update the stream's own state
        // flags, so this mainly catches a stream that was already bad when it
        // was handed to us.
        if (_is.bad()) {
            throw std::runtime_error("I/O error while reading file.");
        }

        // ---- Step 2: parse the text as JSON ----
        // Throws `json::parse_error` on malformed input (including an empty
        // string), which is handled below.
        root = json::parse(content);

        // ---- Step 3: check the root object and required keys ----
        validate_root(root);

        // `at()` is safe here because `validate_root` proved the keys exist.
        // References are used so no JSON data is copied.
        const json& x_json = root.at("x");
        const json& y_json = root.at("y");
        const json& args_json = root.at("args");

        // ---- Step 4: build and validate "x" and "y" ----
        //
        // Validate and populate `x` and `y` before `validate_args` maps `"y"`
        // keys into indices into the hash-sorted `y` vector. "x" has no size
        // limit; "y" is limited so IDs fit in 16 bits.
        x = validate_integer_map(x_json, "x");
        y = validate_integer_map(y_json, "y", max_y_groups);

        // ---- Step 5: "x" and "y" must not share keys ----
        validate_key_separation(x_json, y_json);

        // ---- Step 6: build and validate "args" ----
        // Needs the finished `y` (not `y_json`) to translate keys to indices.
        args = validate_args(args_json, x_json, y_json, y);

        // Reached only if every entry was accepted. There is no separate
        // success flag: `x`, `y` and `args` are now all non-empty, which is
        // what `operator bool()` reports.
        //
        // The early `return` skips the cleanup code after the handlers, which
        // must run ONLY on failure. (A failure always lands in a handler;
        // success returns here.)
        return;

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
    //   ...                -> any non-standard exception
    //
    // Every handler only logs. The members are cleared once, after the
    // handlers, so that no partially-filled state survives a failure.
    // ------------------------------------------------------------------------
    } catch (const json::parse_error& e) {
        std::cerr << "Error: malformed JSON: " << e.what() << '\n';
    } catch (const ValidationError& e) {
        std::cerr << "Error: invalid input: " << e.what() << '\n';
    } catch (const json::exception& e) {
        std::cerr << "Error: JSON processing failure: " << e.what() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
    } catch (...) {
        // Last resort: something that does not derive from std::exception.
        // Catching it is what lets the constructor honour "never throws".
        std::cerr << "Error: unknown non-standard exception.\n";
    }

    // Failure path (the success path returned above): discard anything that
    // was filled in before the error, e.g. `x` and `y` when "args" was bad.
    // After this all three members are empty, so `operator bool()` is false.
    x.clear();
    y.clear();
    args.clear();
}

} // namespace PhantomX
