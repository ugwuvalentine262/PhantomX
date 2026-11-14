#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <string>

#include "indexes.hpp"

namespace PhantomX {

class configuration {

    friend class decision_tree;
    friend class dt_learning;

    struct group {
        std::string name;
        indexes indexes;
    public:
        bool operator<(const group& g) const {
            return name < g.name;
        }
    };

    using groups = std::vector<group>;

    groups x, y, y_map;
    bool okay=false;

public:
    // Boolean type-cast operator returns true iff the configuration
    // was loaded successfully by the constructor.
    operator bool() const;
    // Constructor loads the features configuration from 
    // a JSON file with the specified filename. 
    configuration(const std::string& _filename);
};

using config = configuration;

} // namespace PhantomX

#endif // CONFIGURATION_HPP