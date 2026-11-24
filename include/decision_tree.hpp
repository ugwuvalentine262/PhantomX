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

    struct learner; // forward declaration of the decision-tree learning algorithm.
    struct thinker; // forward declaration of the decision-tree prediction algorithm.

    // An implementation of the node of the decision-tree model.
    struct node {
        std::int32_t left; // index of the left child node in the nodes_ vector.
        std::int32_t right; // index of the right child node in the nodes_ vector.
        std::int32_t index; // index of the feature/attribute used for splitting at this node.
        std::int32_t group; // index of the group of features/attributes used for splitting at this node.
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
};

} // namespace PhantomX

#endif // DECISION_TREE