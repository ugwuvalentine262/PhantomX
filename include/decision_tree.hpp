#ifndef DECISION_TREE
#define DECISION_TREE

#include <cstdint>
#include <istream>
#include <ostream>

#include "configuration.hpp"
#include "indexes.hpp"

namespace PhantomX {

// An implementation of a decision-tree model.
struct decision_tree {

    // The implementation of the final prediction
    // of the decision-tree model.
    struct decision {
        coordinates path; // the decision-path
    };

    // An implementation of the node of the decision-tree model.
    struct node {
        std::int32_t left;
        std::int32_t right;
        std::int32_t index;
        std::int32_t group;
    };

    // An implementation of the leaf-node of the
    // decision-tree model.
    struct leaf {
        coordinates args; // a description of how the target is derived.
        float weight; // the degree of importance of the leaf.
    public:
        // Writes the leaf data to the specified output-stream. 
        // Returns true iff the operation completed successfully.
        bool write(std::ostream& _os) const;
        // Reads the leaf data from the specified input-stream. 
        // Returns true iff the operation completed successfully.
        bool read(std::istream& _is);
    };

    using nodes = std::vector<node>;
    using leafs = std::vector<leaf>;

private:

    // attributes/features configuration containing
    // the grouping of features or attributes.
    const config& config_;

    nodes nodes_; // the nodes of the decision-tree model (excluding leafs).
    leafs leafs_; // the leaf-nodes of the decision-tree model.

public:
    // Writes the nodes and leaf data of the decision-tree
    // to the specified output-stream. Returns true iff the
    // operation completed successfully.
    bool write(std::ostream& _os) const;
    // Reads the nodes and leaf data of the decision-tree
    // from the specified input-stream. Returns true iff the
    // operation completed successfully.
    bool read(std::istream& _is);
    // Clears the decision-tree model.
    void clear();
    // predicts the output given the input attributes.
    decision operator()(const indexes& _attributes) const;
    // Constructor initializes the decision-tree model
    // with the configuration data.
    decision_tree(const config& _config);
    // The default constructor.
    decision_tree()=delete;

};

} // namespace PhantomX

#endif // DECISION_TREE