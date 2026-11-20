#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <istream>
#include <string>

#include "indexes.hpp"

namespace PhantomX {

// Exception type used for all semantic (structural/value) validation failures.
class ValidationError : public std::runtime_error {
public:
    explicit ValidationError(const std::string& message)
        : std::runtime_error(message) {}
};

struct configuration {

    // One entry of "x" or "y": the sorted integers of the array and the
    // std::hash of the corresponding key (the key string is not kept).
    struct group {
        indexes indexes; // sorted ascending, no repeats
        std::size_t hash; // std::hash<std::string> of the key
    public:
        bool operator<(const group& g) const {
            return hash < g.hash;
        }
    };

    using groups = std::vector<group>;
    using args_space = std::vector<indexes>;

    // grouping of x and y attributes/labels
    // into logical features. they are both sorted
    // by hash.
    groups x, y;
    // indices into the sorted y.
    args_space args;
    // this would be set to true if the configurations
    // are loaded successfully by the constructor. 
    bool okay=false;
    
public:
    // Gets a 16-bit ID for the key, which
    // corresponds to a key in y. Returns 0xFFFF if specified
    // key does not exist in y.
    std::uint16_t id(std::size_t _y_key) const;
    // Boolean type-cast operator returns true iff the configuration
    // was loaded successfully by the constructor.
    operator bool() const;
    // Constructor loads the features configuration from 
    // the specified istream object. Reads a JSON document from the istream, 
    // parses it with nlohmann/json and validates it against the following structure:
    //
    //   {
    //     "x":    { "<key>": [non-negative integers, at least one], ... },
    //     "y":    { "<key>": [non-negative integers, at least one], ... },
    //     "args": [ ["<key in y>", ...], ... ]
    //   }
    //
    // Extra root-level keys are allowed and ignored.
    //
    // Integers must lie in [0, INT_MAX] and must not repeat within one array.
    // "x" and "y" are stored as std::vector<Sample> sorted by key hash (the key
    // string itself is discarded); each Sample's integers are sorted ascending.
    // Hash collisions between keys of the same object are rejected. "args" is
    // stored as std::vector<std::vector<int>> holding indices into the sorted
    // "y" vector. The stored data is not used further.
    configuration(std::istream& _is);
    // Deleted default construction.
    configuration()=delete;

};

using config = configuration;

} // namespace PhantomX

#endif // CONFIGURATION_HPP