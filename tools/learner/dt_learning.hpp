#ifndef DECISION_TREE_LEARNING
#define DECISION_TREE_LEARNING

#include <algorithm>
#include <span>

#include <decision_tree.hpp>
#include <dataset.hpp>

namespace PhantomX {

struct dt_learning {

    using branch = indexes;

    struct split {
        int score_=-1;
        branch left_, right_;
        int index_, group_;
    public:
        operator bool() const;
        bool operator>(const split& _split) const;
    };

    struct state {
        std::vector<groups> online_grouping_;
    public:
        void apply(const dataset& _dataset, const split& _split, const indexes& _samples);
        void undo(const split& _split, const indexes& _samples);

    };

    const dataset& dataset_;
    groups groups_;
    decision_tree::nodes nodes_;
    decision_tree::leafs leafs_;
    size_t node_idx_, leaf_idx_;
    int num_of_nodes_;
    int num_of_leafs_;
    int min_features_;
    int min_samples_leaf_;
    int min_samples_split_;
    int max_depth_;

private:
    split divide(const indexes& _samples, const state& _state, const group& _group, int _depth) const;
    bool valid(const split& _split) const;
    int recursive_greedy_splitting(state& _state, const split& _split, const indexes& _samples, int _depth=0);
public:

    decision_tree operator()(const indexes& _samples) {
        split null_split;
        state state;
        (void)recursive_greedy_splitting(state, null_split, _samples);
        return decision_tree(dataset_.groups_, std::move(nodes_), std::move(leafs_));
    }

    dt_learning(const dataset& _dataset, int _min_features, int _min_samples_leaf, int _min_samples_split);

};

} // namespace PhantomX

#endif // DECISION_TREE_LEARNING