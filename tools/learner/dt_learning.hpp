#ifndef DECISION_TREE_LEARNING
#define DECISION_TREE_LEARNING

#include <algorithm>
#include <span>

#include <decision_tree.hpp>
#include <configuration.hpp>
#include <dataset.hpp>

namespace PhantomX {

// This class contains our implementation of a decision-tree 
// learning algorithm.
struct dt_learning {

    // A branch resulting from a split during the recursive greedy
    // splitting routine.
    struct branch { indexes samples; bool true_branch; };

    // The split structure contains information necessary
    // for splitting samples at a node during the recursive-
    // greedy-splitting routine.
    struct split {
        branch left, right; // the left and right branches of the split.
        int score=-1; // a non-negative score reflecting the quality of the split.
        int index, group;
        bool negate;
    public:
        // Type-cast operator returns true iff the split is valid.
        // i.e, if the score is non-negative.
        operator bool() const;
        // Comparison operator returns true iff the score of the
        // split is greater than that of the specified one.
        bool operator>(const split& _split) const;
    };

    // The state structure holds info necessary traversing
    // the tree-structure of the recursive-greedy-splitting algorithm.
    struct state {

        struct group {
            // the attributes that make up the group.
            indexes indices;
            // a group is a leaf iff no other group has resulted
            // from an intersection of this group with another.
            bool leaf;
        public:
            // Constructor.
            // The new group is formed from the intersection of the specified
            // groups. The newly created group is initially a leaf.
            group(const config::group& _a, group& _b);
            // Constructor.
            // Initializes the group with indexes.
            // The newly created group is initially a leaf.
            group(const indexes& _indexes);
        };

        using decision_path = std::vector<group>;

        // keeping track of all decisions made on all
        // samples of the dataset by the decision-tree
        // learning algorithm.
        std::vector<decision_path> paths;
    public:
        // Updates the current state with respect to the given parameters.
        void apply(const config& _config, const split& _split, const branch& _branch);
        // Retracts the split that was applied to the state.
        // The state returns to the exact state it was before the
        // split was applied.
        void undo(const split& _split, const branch& _branch);
        // Constructor initializes the state with a dataset.
        state(const dataset& _dataset);
    };

    // a read-only reference to the dataset used for
    // training the decision-tree model.
    const dataset& dataset_;
    // features grouping/configuration data.
    const config& config_;

    // the nodes of the decision-tree resulting from
    // the learning algorithm.
    decision_tree::nodes nodes_;
    // the leafs of the decision-tree resulting
    // from the learning algorithm.
    decision_tree::leafs leafs_;

    // Number of nodes in the tree (including leafs).
    int node_idx_=0;
    // The number of pure leaf nodes in the tree.
    int num_of_pure_leafs_=0;

    /* ----------- Hyper-parameters --------------- */

    // Number of features considered while 
    // looking for a split point.
    int max_features_;
    // The minimum number of samples that is expected
    // to be at a leaf.
    int min_samples_leaf_;
    // The minimum number of samples required before
    // considering splitting that node.
    int min_samples_split_;

private:
    // Splits the samples using the specified feature/group, and returns the resulting split data.
    split divide(const indexes& _samples, const state& _state, const config::group& _group, int _index, bool _flip) const;
    // The recursive greedy splitting algorithm is the core of the decision tree learning algorithm.
    // This is a process used by the decision tree to divide a dataset into smaller subsets based on the 
    // features that best separate them into meaningful clusters.
    int recursive_greedy_splitting(state& _state, const split& _split, const branch& _branch);
    // Adds a node to the decision-tree model.
    void add_node(int _node_idx, int _right_node_idx, int _left_node_idx, const split& _split);
    // Adds a leaf node to the decision-tree model.
    void add_leaf(int _node_idx);

public:
    // Calls the main learning algorithm for the decision-tree model.
    void operator()(indexes _samples);
    // Constructor.
    // The learning algorithm is initialized with a dataset, a configuration/grouping of the
    // attributes and features, then some hyperparameter determining the structure of the decision-tree.
    dt_learning(const dataset& _dataset, const config& _config, int _max_features, int _min_samples_leaf, int _min_samples_split);
};

} // namespace PhantomX

#endif // DECISION_TREE_LEARNING