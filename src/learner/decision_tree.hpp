// ============================================================================
//  decision_tree.hpp  (learner section)
// ============================================================================
//
//  Declares `decision_tree::learner`, the class that trains a decision-tree
//  model from a dataset using recursive greedy splitting.
//
//  Overview of how the pieces fit together:
//
//    1. The caller constructs a `learner` with a dataset and a feature
//       grouping/configuration, optionally tunes hyper-parameters through
//       the set_* functions, and then calls learn() with the indexes of the
//       training samples.
//
//    2. learn() runs recursive_greedy_splitting(). At each node, divide() is
//       tried on candidate features/groups, and the best resulting `split`
//       (a left `branch` and a right `branch`) is chosen.
//
//    3. A `state` object tracks which groups every sample has passed through.
//       It is updated with apply() when a split is explored and rolled back
//       with undo() afterwards (classic backtracking).
//
//    4. When a node is finished, add_node() or add_leaf() records it in
//       nodes_ / leafs_, which together form the resulting tree. At the end,
//       learn() moves them into the `decision_tree` object it returns.
//
// ============================================================================

#ifndef DECISION_TREE_LEARNER
#define DECISION_TREE_LEARNER

#include <decision_tree.hpp>  // declares `decision_tree` (nodes, leafs, ...)
#include <configuration.hpp>  // declares `config` (feature grouping data)
#include <dataset.hpp>        // declares `dataset`, `indexes`, `coordinates`

namespace PhantomX {

// ----------------------------------------------------------------------------
// decision_tree::learner
//
// Our implementation of a decision-tree learning algorithm. It is declared as
// a nested type of `decision_tree`, so it can fill in the tree's internal
// containers (nodes and leafs) directly.
// ----------------------------------------------------------------------------
struct decision_tree::learner {

    // ------------------------------------------------------------------------
    // branch
    //
    // One side of a split, produced during the recursive greedy splitting
    // routine.
    // ------------------------------------------------------------------------
    struct branch {
        // The indexes of the samples that fall into this branch.
        indexes samples;

        // A description of how the target value is derived for this branch
        // (e.g. the parameters later used to predict at the corresponding
        // leaf).
        //
        // It is the selection returned by Objective::solve(), as a list of
        // coordinates. In each coordinate, `x` is the position, in the
        // decision path, of a leaf group, and `y` is the index of the target
        // assigned to that group. For the left branch, positions refer to the
        // path after the split is applied (the new group is at the end).
        coordinates args;

        // The score of this branch with respect to some objective function.
        // -1 means "not yet scored".
        int score = -1;

        // True iff this is the "true" branch of the split, which corresponds
        // to the left branch. (Left uninitialized: set it before reading.)
        bool true_branch;
    };

    // ------------------------------------------------------------------------
    // split
    //
    // Everything needed to split the samples at a node during the recursive
    // greedy splitting routine.
    // ------------------------------------------------------------------------
    struct split {
        // The two branches that result from the split.
        branch left, right;

        // A non-negative score reflecting the quality of the split.
        // -1 means "not yet scored".
        int score = -1;

        // What the split was made on: `group` is the feature group
        // (config_.x[group]) and `index` is the position, in the decision
        // path, of the group that is intersected with it.
        // (Left uninitialized: set before reading.)
        int index, group;
    };

    // ------------------------------------------------------------------------
    // state
    //
    // Holds the information needed to traverse the tree structure built by the
    // recursive greedy splitting algorithm. It supports apply()/undo() so the
    // algorithm can try a split and then roll it back.
    // ------------------------------------------------------------------------
    struct state {

        // --------------------------------------------------------------------
        // group
        //
        // A set of attributes (features) that belong together.
        // --------------------------------------------------------------------
        struct group {
            // Position (in the same decision_path) of the parent group this
            // group was formed from, set ONLY when that parent was a leaf at
            // the time of creation; otherwise -1. undo() uses it to make the
            // parent a leaf again. A position stays valid when the
            // decision_path vector reallocates, unlike a pointer.
            // (Declared first because both constructors initialize it first.)
            int parent;

            // The attributes that make up the group.
            indexes indices;

            // A group is a leaf iff no other group has resulted from an
            // intersection of this group with another one.
            bool leaf;

        public:
            // Constructor.
            // Forms a new group from the intersection of the specified
            // groups. The newly created group is initially a leaf and `_b` is
            // marked as no longer being a leaf. `_b_index` is the position of
            // `_b` in the decision path; `parent` is set to it only if `_b`
            // was a leaf (-1 otherwise).
            group(const config::group& _a, group& _b, int _b_index);

            // Constructor.
            // Initializes the group from a set of indexes.
            // The newly created group is initially a leaf and has no parent
            // (`parent` is -1).
            group(const indexes& _indexes);
        };

        // The ordered list of groups a single sample has passed through,
        // i.e. the decisions made on that sample from the root downwards.
        using decision_path = std::vector<group>;

        // Keeps track of all decisions made on all samples of the dataset by
        // the decision-tree learning algorithm (one path per sample).
        std::vector<decision_path> paths;

    public:
        // Updates the current state with respect to the given parameters:
        // records the effect of applying `_split`, for the samples in
        // `_branch`.
        void apply(const config& _config, const split& _split, const branch& _branch);

        // Retracts a split that was previously applied to the state. The
        // state returns to exactly what it was before the split was applied.
        void undo(const split& _split, const branch& _branch);

        // Constructor.
        // Initializes the state from a dataset.
        state(const dataset& _dataset);
    };

    // ------------------------------------------------------------------------
    // Inputs
    // ------------------------------------------------------------------------

    // Read-only reference to the dataset used to train the model.
    // The dataset must outlive the learner.
    const dataset& dataset_;

    // Feature grouping/configuration data (read-only reference).
    // The config must outlive the learner.
    const config& config_;

    // ------------------------------------------------------------------------
    // Working storage for the tree being built by the learning algorithm.
    // learn() moves these into the decision_tree it returns.
    // ------------------------------------------------------------------------

    // The (internal) nodes of the tree being built (excluding leaves).
    decision_tree::nodes nodes_;

    // The leaves of the tree being built.
    decision_tree::leafs leafs_;

    // Number of nodes in the tree so far (including leaves). Presumably also
    // used as the index handed to the next node that is created.
    int node_idx_ = 0;

    // ------------------------------------------------------------------------
    // Hyper-parameters
    //
    // The three integer constraints below (max_features_, min_samples_leaf_,
    // min_samples_split_) all follow the same convention: a NEGATIVE value
    // means the constraint is IGNORED (disabled). Each defaults to -1, so by
    // default none of them restricts the learning algorithm. Use the
    // corresponding set_* function to enable one.
    //
    // growth_ is different: it is a real-valued constant that must be
    // strictly positive (see below).
    // ------------------------------------------------------------------------

    // Number of features considered while looking for a split point.
    // Negative (default -1) = no limit; all features are considered.
    int max_features_ = -1;

    // The minimum number of samples expected to be at a leaf.
    // Negative (default -1) = no minimum is enforced on leaf size.
    int min_samples_leaf_ = -1;

    // The minimum number of samples a node must contain before splitting it
    // is even considered.
    // Negative (default -1) = nodes are considered for splitting regardless
    // of how few samples they contain.
    int min_samples_split_ = -1;

    // Exponential growth constant `k`, used to set the weight of the leaf
    // nodes of the decision tree via the function:
    //
    //     f(x) = e^(k * x) - 1,     where x is in (0, 1]
    //
    // Here `x` is the purity of the node (higher x = purer node), so a leaf
    // that is purer receives a larger weight. A larger `k` makes the weight
    // grow more steeply as purity increases; as k approaches 0 the weight
    // approaches 0 for every leaf.
    //
    // Must be strictly greater than 0 (enforced by set_growth_rate()).
    // Default: 2.5.
    float growth_ = 2.5;

private:
    // ------------------------------------------------------------------------
    // Internal algorithm steps
    // ------------------------------------------------------------------------

    // Splits the samples using the specified feature/group (`_group`, `_index`)
    // and returns the resulting split data (both branches and their scores).
    // Does not modify the learner, hence const.
    split divide(const indexes& _samples,
                 const state& _state,
                 int _group,
                 int _index) const;

    // The recursive greedy splitting algorithm: the core of the decision-tree
    // learning algorithm. It divides a dataset into smaller subsets based on
    // the features that best separate them into meaningful clusters, then
    // recurses into each subset.
    //
    // Parameters:
    //   _state  - tracks the decisions made on all samples so far; modified
    //             via apply() and restored via undo() while exploring splits.
    //   _split  - the split that produced the branch being processed.
    //   _branch - the branch (and its samples) this call is responsible for.
    //
    // Returns: a non-negative index identifying the node this call was made
    // on. This index is its position in the vector the nodes are serialized
    // into, and it is what allows a parent to know where both of its
    // children are located in that vector (see add_node()).
    int recursive_greedy_splitting(state& _state,
                                   const split& _split,
                                   const branch& _branch);

    // Adds an internal node to the decision-tree model, linking it to its
    // left and right children (given by node index).
    void add_node(int _node_idx, int _left_node, int _right_node, const split& _split);

    // Adds a leaf node to the decision-tree model.
    void add_leaf(int _node_idx, const branch& _branch);

public:
    // ------------------------------------------------------------------------
    // Hyper-parameter setters
    //
    // Each setter returns true if the argument is accepted and false if it
    // is rejected; the acceptance rule differs between the integer setters
    // and the growth-rate setter (see below).
    // ------------------------------------------------------------------------

    // Sets max_features_.
    // Returns true iff `i` is non-zero, otherwise false.
    // NOTE: a negative `i` is non-zero, so the call returns true, but a
    // negative value means the constraint is IGNORED (no limit on the number
    // of features considered at each split).
    bool set_max_features(int i);

    // Sets min_samples_leaf_.
    // Returns true iff `i` is non-zero, otherwise false.
    // NOTE: a negative `i` returns true, but disables the constraint (no
    // minimum leaf size is enforced).
    bool set_min_samples_leaf(int i);

    // Sets min_samples_split_.
    // Returns true iff `i` is non-zero, otherwise false.
    // NOTE: a negative `i` returns true, but disables the constraint (nodes
    // are considered for splitting regardless of their sample count).
    bool set_min_samples_split(int i);

    // Sets growth_, the exponential growth constant `k` in
    // f(x) = e^(k * x) - 1 used to weight leaf nodes by purity.
    // Returns true iff `v` is strictly greater than 0, otherwise false.
    bool set_growth_rate(float v);

    // ------------------------------------------------------------------------
    // Public interface
    // ------------------------------------------------------------------------

    // Runs the main learning algorithm for the decision-tree model on the
    // samples given by `_samples` (indexes into the dataset) and returns the
    // learned decision tree.
    //
    // The tree is built in nodes_ and leafs_, which are then moved into the
    // returned object, so the learner holds no tree afterwards. Calling
    // learn() again starts from scratch; the hyper-parameters are kept.
    decision_tree learn(indexes _samples);

    // Constructor.
    // Initializes the learner with a dataset and a configuration/grouping of
    // the attributes and features.
    learner(const dataset& _dataset, const config& _config);
};

} // namespace PhantomX

#endif // DECISION_TREE_LEARNER
