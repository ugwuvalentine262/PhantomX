// ============================================================================
//  configuration.hpp
// ============================================================================
//
//  Declares `configuration` (aliased as `config`), the feature-grouping
//  data that drives both learning and prediction in PhantomX.
//
//  Role of this type:
//
//    - `configuration` describes HOW raw features ("x") and target labels
//      ("y") are grouped into logical features. A group is a set of
//      attribute indices that are treated as one unit when searching for
//      splits (see `decision_tree::learner::divide()`).
//
//    - It is loaded from a JSON document once, validated, and then held as
//      read-only reference data by learners and thinkers. It carries no
//      learned parameters.
//
//    - `args` names, for each logical feature group in "y", the ordered
//      sequence of "y" keys used to derive the target. It is stored purely
//      as indices into the sorted "y" vector; the key strings themselves
//      are discarded after loading.
//
//  Key design points:
//
//    - Groups are stored SORTED BY HASH, not by key string. The hash is a
//      64-bit FNV-1a hash of the key (implemented in configuration.cpp),
//      which is fully specified and therefore identical on every platform
//      and in every run.
//      The key string is not retained at all. This makes lookup and equality
//      cheap at the cost of not being able to reconstruct the original keys.
//
//    - Index arrays within a group are sorted ascending and contain no
//      repeats; this is enforced at load time.
//
//    - Hash collisions between distinct keys of the same object are
//      REJECTED at load time rather than silently merged.
//
//    - "y" holds at most `max_y_groups` (65536) groups, so that every
//      position in the sorted "y" vector fits in the 16-bit ID returned by
//      `id()`. This is enforced at load time.
//
//    - "args" holds at most `max_args` (8) inner arrays. This is enforced
//      at load time.
//
//    - There is no separate "loaded successfully" flag. A configuration is
//      valid exactly when "x", "y" and "args" are all non-empty (see
//      `operator bool()`). A successful load always yields three non-empty
//      members; a failed load always yields three empty ones.
//
//    - The constructor NEVER throws. Every failure is caught inside the
//      constructor, reported on `std::cerr`, and results in a cleared
//      (invalid) configuration.
//
// ============================================================================

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <cstddef>
#include <cstdint>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>
#include <optional>

#include <indexes.hpp>

namespace PhantomX {

// ----------------------------------------------------------------------------
// ValidationError
//
// Exception type used for all semantic (structural/value) validation
// failures during configuration loading. Derives from std::runtime_error so
// callers that already catch std::runtime_error do not need to change.
//
// Note: it is thrown and caught INSIDE the `configuration` constructor; it
// never propagates out of it. It is exposed so that the validation helpers
// in configuration.cpp (and any tests of them) can share one type.
// ----------------------------------------------------------------------------
class ValidationError : public std::runtime_error {
public:

    // Constructs a ValidationError carrying the specified human-readable
    // message.
    explicit ValidationError(const std::string& message)
        : std::runtime_error(message) {}
};

// ----------------------------------------------------------------------------
// configuration
//
// Holds the feature grouping of "x" (features) and "y" (labels) attributes,
// plus the `args` table that maps each logical target group to a sequence of
// "y" keys.
// ----------------------------------------------------------------------------
struct configuration {

    // ------------------------------------------------------------------------
    // group
    //
    // One entry of "x" or "y": the sorted integers of the array and the
    // 64-bit FNV-1a hash of the corresponding key (the key string itself is
    // not kept, to save space and because only the hash and indices are
    // needed downstream).
    //
    // Portability note: the hash is a standard 64-bit FNV-1a (offset basis
    // 14695981039346656037, prime 1099511628211, bytes read as unsigned
    // char), implemented privately in configuration.cpp. It is fully
    // specified and uses a fixed width, so the same key always produces the
    // same hash on every platform and in every run.
    // Group order, and therefore the IDs returned by `id()`, depends only on
    // the set of keys. IDs are still NOT stable across configurations that
    // have different key sets.
    // ------------------------------------------------------------------------
    struct group {

        // The attribute/label indices that make up this group.
        // Sorted ascending, contains no repeats.
        indexes indexes;

        // 64-bit FNV-1a hash of the group's key (see above). Used for
        // ordering and for equality/lookup; the key string itself is
        // discarded.
        std::uint64_t hash;

    public:

        // Orders groups by hash. This is the ordering used to keep
        // `configuration::x` / `configuration::y` sorted, which in turn is
        // what makes `id()` a binary search.
        bool operator<(const group& g) const {
            return hash < g.hash;
        }
    };

    // ------------------------------------------------------------------------
    // Public type aliases
    // ------------------------------------------------------------------------

    // The sorted collection of feature groups.
    using groups = std::vector<group>;

    // The `args` table: for each logical target group, the ordered list of
    // indices into the sorted "y" vector that derive that target.
    using args_space = std::vector<indexes>;

    // ------------------------------------------------------------------------
    // Limits
    // ------------------------------------------------------------------------

    // Maximum number of keys (groups) allowed in "y". Chosen so that every
    // position in the sorted "y" vector (0 .. 65535) fits in the
    // `std::uint16_t` returned by `id()`.
    static constexpr std::size_t max_y_groups = 65536;

    // Maximum number of inner arrays allowed in "args".
    static constexpr std::size_t max_args = 8;

    // ------------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------------

    // Grouping of "x" and "y" attributes/labels into logical features.
    // Both are sorted by hash. Both are empty if loading failed.
    groups x, y;

    // Indices into the sorted `y`. Empty if loading failed.
    args_space args;

public:

    // ------------------------------------------------------------------------
    // Lookup
    // ------------------------------------------------------------------------

    // Gets the 16-bit ID of the key `_key` in `y`, or std::nullopt if `y`
    // has no such key.
    //
    // The key is hashed with the same 64-bit FNV-1a hash that is stored in
    // `group::hash` (an implementation detail of configuration.cpp) and
    // looked up by binary search in the
    // hash-sorted `y` vector. Since the key strings are not retained, a
    // key is identified by its hash alone; collisions within "y" are
    // rejected at load time, so the match is unambiguous within a given
    // configuration. Callers are expected to pass only keys that are known
    // to belong to "y" (the nullopt result is a safety net, not a way to
    // test membership of arbitrary strings).
    //
    // The ID is the group's position in the hash-sorted `y` vector. It is
    // stable for a given configuration but NOT across configurations
    // (inserting a key can renumber existing keys). Because `y` is limited
    // to `max_y_groups` entries at load time, the position always fits in
    // 16 bits.
    //
    // Returns std::nullopt if the key is not present (which includes the
    // case of an invalid, empty configuration). Never throws.
    std::optional<std::uint16_t> id(const std::string& _key) const noexcept;

    // ------------------------------------------------------------------------
    // Validity
    // ------------------------------------------------------------------------

    // Explicit boolean conversion. Returns true iff `x`, `y` and `args` are
    // ALL non-empty; returns false if any of them is empty.
    //
    // Since a successful load always produces three non-empty members, and a
    // failed load clears all three, this is equivalent to "the constructor
    // succeeded". Being `explicit`, it works in conditions (`if (cfg)`,
    // `!cfg`) and with static_cast<bool>, but does not silently convert to
    // bool/int in other contexts.
    //
    // Never throws.
    explicit operator bool() const noexcept;

    // ------------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------------

    // Constructor.
    //
    // Loads the features configuration from the specified istream object.
    // Reads a JSON document from the istream, parses it with nlohmann/json,
    // and validates it against the following structure:
    //
    //   {
    //     "x":    { "<key>": [non-negative integers, at least one], ... },
    //     "y":    { "<key>": [non-negative integers, at least one], ... },
    //     "args": [ ["<key in y>", ...], ... ]
    //   }
    //
    // Extra root-level keys are allowed and ignored.
    //
    // Validation rules:
    //
    //   - The root is a JSON object containing "x", "y" and "args".
    //   - "x" and "y" are non-empty JSON objects. Each value is a non-empty
    //     array of integers.
    //   - Integers must lie in [0, INT_MAX] and must not repeat within one
    //     array. Booleans, null, strings and floating-point numbers are
    //     rejected.
    //   - "x" and "y" are stored as `groups` sorted by key hash; the key
    //     string itself is discarded and each group's integers are sorted
    //     ascending.
    //   - "y" has at most `max_y_groups` (65536) keys.
    //   - Hash collisions (equal 64-bit FNV-1a hashes) between keys of the
    //     same object are rejected.
    //   - "x" and "y" share no keys (they are mutually exclusive).
    //   - "args" is a non-empty array with at most `max_args` (8) elements.
    //     Each element is a non-empty array of strings; every string must be
    //     a key of "y" (a key of "x" is reported with a specific message),
    //     and a string must not repeat within one inner array (repeats
    //     across different inner arrays are allowed).
    //   - "args" is stored as `args_space` holding indices into the sorted
    //     "y" vector, in the order written. (Its stored data is not used
    //     further by this class.)
    //
    // Not detected: a key repeated within one JSON object. The JSON parser
    // keeps only the last occurrence, so the earlier ones are silently
    // dropped before validation sees them.
    //
    // Error handling:
    //
    //   - This constructor never throws. All failures are caught inside it:
    //     I/O errors, malformed JSON, validation failures (ValidationError),
    //     any other nlohmann::json or std::exception, and any non-standard
    //     exception.
    //   - Each failure writes one diagnostic line to `std::cerr`.
    //   - On failure, `x`, `y` and `args` are all cleared, so
    //     `operator bool()` returns false.
    configuration(std::istream& _is);

    // Default construction is deleted: a configuration must always be
    // constructed from a JSON document. (Note that a configuration built
    // from an invalid document still exists, but is empty and tests false;
    // check `operator bool()` after construction.)
    configuration() = delete;
};

// Short alias used throughout the library.
using config = configuration;

} // namespace PhantomX

#endif // CONFIGURATION_HPP
