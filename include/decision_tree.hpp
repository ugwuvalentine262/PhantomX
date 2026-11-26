// ============================================================================
//  decision_tree.hpp
// ============================================================================
//
//  Declares `decision_tree`, the container type for a single trained
//  decision-tree model.
//
//  Role of this type:
//
//    - `decision_tree` is a *model* type. It holds the trained tree
//      structure (internal split nodes + leaf nodes) and knows how to
//      serialize itself. It does not train and it does not predict.
//
//    - Training is delegated to the nested `decision_tree::learner`, and
//      prediction is delegated to the nested `decision_tree::thinker`. Both
//      are forward declared here and defined in their own headers so this
//      header stays lightweight.
//
//    - Because `learner` and `thinker` are nested types of `decision_tree`,
//      they have access to the private `nodes_` / `leafs_` containers and
//      can populate / read them directly without extra accessors.
//
//  Tree representation:
//
//    The tree is stored as two parallel flat vectors:
//
//      - `nodes_` holds the internal (splitting) nodes. Each node names its
//        left and right children by index and describes the split
//        (feature index + feature group index).
//
//      - `leafs_` holds the terminal (leaf) nodes. Each leaf carries a
//        `coordinates` descriptor (how the target value is derived there)
//        and a `weight` (the leaf's importance, e.g. purity-based).
//
//    Child references in `node` are signed so that a negative value can be
//    used to mean "this child is a leaf" if the convention is adopted; the
//    exact child-encoding convention is decided by `learner` / `thinker`.
//
// ============================================================================

#ifndef DECISION_TREE
#define DECISION_TREE

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include <configuration.hpp>
#include <indexes.hpp>

namespace PhantomX {

// ----------------------------------------------------------------------------
// decision_tree
//
// An implementation of a decision-tree model.
// ----------------------------------------------------------------------------
struct decision_tree {

    // ------------------------------------------------------------------------
    // Nested algorithm types (forward declarations)
    // ------------------------------------------------------------------------

    // The decision-tree learning algorithm. Defined in its own header.
    // Fills in `nodes_` / `leafs_` when training.
    struct learner;

    // The decision-tree prediction algorithm. Defined in its own header.
    // Reads `nodes_` / `leafs_` when predicting.
    struct thinker;

    // ------------------------------------------------------------------------
    // node
    //
    // An implementation of an internal (splitting) node of the decision-tree
    // model. A node performs a comparison on one feature and routes each
    // sample to either its left or its right child.
    // ------------------------------------------------------------------------
    struct node {

        // Index of the left child node in the `nodes_` vector.
        std::int32_t left;

        // Index of the right child node in the `nodes_` vector.
        std::int32_t right;

        // Index of the feature/attribute used for splitting at this node.
        std::int32_t index;

        // Index of the group of features/attributes used for splitting at
        // this node (see `config` for how features are grouped).
        std::int32_t group;
    };

    // ------------------------------------------------------------------------
    // leaf
    //
    // An implementation of a leaf (terminal) node of the decision-tree
    // model. A leaf produces the tree's output for any sample routed to it.
    // ------------------------------------------------------------------------
    struct leaf {

        // A description of how the target is derived at this leaf.
        coordinates args;

        // The degree of importance of the leaf. Typically the purity-based
        // weight computed by the learner (e.g. via the `growth` constant).
        float weight;

    public:

        // Writes the leaf data to the specified output-stream.
        // Returns true iff the operation completed successfully.
        bool write(std::ostream& _os) const;

        // Reads the leaf data from the specified input-stream.
        // Returns true iff the operation completed successfully.
        bool read(std::istream& _is);
    };

    // ------------------------------------------------------------------------
    // Public type aliases
    // ------------------------------------------------------------------------

    // The collection of internal (splitting) nodes of a tree.
    using nodes = std::vector<node>;

    // The collection of leaf nodes of a tree.
    using leafs = std::vector<leaf>;

private:

    // ------------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------------

    // The internal nodes of the decision-tree model (excluding leaves).
    nodes nodes_;

    // The leaf nodes of the decision-tree model.
    leafs leafs_;

public:

    // ------------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------------

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