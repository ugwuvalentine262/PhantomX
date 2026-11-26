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
//    - Groups are stored SORTED BY HASH, not by key string. The key string
//      is not retained at all. This makes lookup and equality cheap at the
//      cost of not being able to reconstruct the original keys.
//
//    - Index arrays within a group are sorted ascending and contain no
//      repeats; this is enforced at load time.
//
//    - Hash collisions between distinct keys of the same object are
//      REJECTED at load time rather than silently merged.
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
    // std::hash of the corresponding key (the key string itself is not
    // kept, to save space and because only the hash and indices are needed
    // downstream).
    // ------------------------------------------------------------------------
    struct group {

        // The attribute/label indices that make up this group.
        // Sorted ascending, contains no repeats.
        indexes indexes;

        // std::hash<std::string> of the group's key. Used for ordering and
        // for equality/lookup; the key string itself is discarded.
        std::size_t hash;

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
    // State
    // ------------------------------------------------------------------------

    // Grouping of "x" and "y" attributes/labels into logical features.
    // Both are sorted by hash.
    groups x, y;

    // Indices into the sorted `y`.
    args_space args;

    // True iff the configurations were loaded successfully by the
    // constructor. Mirrored by `operator bool()`.
    bool okay = false;

public:

    // ------------------------------------------------------------------------
    // Lookup
    // ------------------------------------------------------------------------

    // Gets the 16-bit ID of the key with the specified hash, or std::nullopt
    // if no key in `y` has that hash.
    //
    // The ID is the group's position in the hash-sorted `y` vector. It is
    // stable for a given configuration but NOT across configurations
    // (inserting a key can renumber existing keys).
    //
    // Returns std::nullopt if the hash is not present.
    std::optional<std::uint16_t> id(std::size_t _y_key) const;

    // ------------------------------------------------------------------------
    // Validity
    // ------------------------------------------------------------------------

    // Boolean type-cast operator. Returns true iff the configuration was
    // loaded successfully by the constructor. Equivalent to reading `okay`.
    operator bool() const;

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
    //   - Integers must lie in [0, INT_MAX] and must not repeat within one
    //     array.
    //   - "x" and "y" are stored as `groups` sorted by key hash; the key
    //     string itself is discarded and each group's integers are sorted
    //     ascending.
    //   - Hash collisions between keys of the same object are rejected.
    //   - "args" is stored as `args_space` holding indices into the sorted
    //     "y" vector. (Its stored data is not used further by this class.)
    //
    // If validation fails, `okay` remains false and `operator bool()`
    // returns false. (Whether the constructor throws `ValidationError` on
    // failure or merely leaves `okay` false is decided by the
    // implementation in configuration.cpp; this header documents both
    // signals as present.)
    configuration(std::istream& _is);

    // Default construction is deleted: a configuration must always come
    // from a validated JSON document, so an "empty" instance is never a
    // legal state to have.
    configuration() = delete;
};

// Short alias used throughout the library.
using config = configuration;

} // namespace PhantomX

#endif // CONFIGURATION_HPP