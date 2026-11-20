#include <algorithm>
#include <random>
#include <numeric>

#include "dt_learning.hpp"
#include "objective.hpp"

namespace PhantomX {

    // The recursive greedy splitting algorithm is the core of the decision tree learning algorithm.
    // This is a process used by the decision tree to divide a dataset into smaller subsets based on the 
    // features that best separate them into meaningful clusters.
    int dt_learning::recursive_greedy_splitting(state& _state, const split& _split, const branch& _branch) {
        int node_idx = node_idx_++;
        split best_split;
        _state.apply(config_, _split, _branch);

        for (auto& g : config_.x) {
            for (size_t i=0; i<=_state.paths[0].size(); i++) {
                for (bool flip : {false, true}) {
                    split split = divide(_branch.samples, _state, g, i, flip);
                    if (split) {
                        if (!best_split || (split > best_split)) {
                            best_split = std::move(split);
                        }
                    }
                }
            }
        }
        // did we find a good split?
        if (best_split) {
            add_node(   node_idx
                    ,   recursive_greedy_splitting(_state, best_split, best_split.left)
                    ,   recursive_greedy_splitting(_state, best_split, best_split.right)
                    ,   best_split
                );
        } else { // no good split was found.
            add_leaf(node_idx);
        }
        _state.undo(_split, _branch);
        return node_idx;
    }

    // Adds a node to the decision-tree model.
    void dt_learning::add_node(int _node_idx, int _right_node_idx, int _left_node_idx, const split& _split) {

    }

    // Adds a leaf node to the decision-tree model.
    void dt_learning::add_leaf(int _node_idx) {

    }

    // Type-cast operator returns true iff the split is valid.
    // i.e, if the score is non-negative.
    dt_learning::split::operator bool() const {
        return score<0?true:false;
    }

    // Comparison operator returns true iff the score of the
    // split is greater than that of the specified one.
    bool dt_learning::split::operator>(const split& _split) const {
        return score > _split.score;
    }

    // Updates the current state with respect to the given parameters.
    void dt_learning::state::apply(const config& _config, const dt_learning::split& _split, const branch& _branch)
    {
        if (!_split||!_branch.true_branch) { return; }
        for (auto& i : _branch.samples) {
            auto& p = paths.at(i);
            auto& a = _config.x.at(_split.group);
            auto& b = p.at(_split.index);

            p.emplace_back(a, b);
        }
    }

    // Retracts the split that was applied to the state.
    // The state returns to the exact state it was before the
    // split was applied.
    void dt_learning::state::undo(const dt_learning::split& _split, const branch& _branch) {
        if (!_split||!_branch.true_branch) { return; }
        for (size_t i=0; i < _branch.samples.size(); i++) {
            paths[_branch.samples[i]].pop_back();
        }
    }

    // Constructor initializes the state with a dataset.
    dt_learning::state::state(const dataset& _dataset) {
        paths.reserve(_dataset.size());
        for (auto& sample : _dataset.samples) {
            paths.emplace_back(decision_path{group(sample.x)});
        }
    }

    // Constructor.
    // The new group is formed from the intersection of the specified
    // groups. The newly created group is initially a leaf.
    dt_learning::state::group::group(const config::group& _a, dt_learning::state::group& _b)
        :   group(_a.indexes & _b.indices)
    {
        _b.leaf=false;
    }

    // Constructor.
    // Initializes the group with indexes.
    // The newly created group is initially a leaf.
    dt_learning::state::group::group(const indexes& _indexes)
        :   indices(_indexes)
        ,   leaf(true)
    {}

    // Splits the samples using the specified feature/group, and returns the resulting split data.
    dt_learning::split dt_learning::divide(const indexes& _samples, const state& _state, const config::group& _group, int _index, bool _flip) const {
        split split;
        Objective::Matrix left, right;
        auto& first = _state.paths.at(_samples.front());

        int cnt = std::count_if(first.begin(), first.end(), [](const state::group& g)
                        {
                            return g.leaf;
                        });

        int llc=cnt, rlc=cnt;

        first[_index].leaf?rlc:++llc;

        for (auto sample : _samples) {
            auto& path = _state.paths[sample];

            path[_index]
        }

        auto right_res = Objective::solve(right, );
        auto left_res = Objective::solve(left, );

        return split;
    }

    // Calls the main learning algorithm for the decision-tree model.
    void dt_learning::operator()(indexes _samples) {
        split null_split;
        state state(dataset_);
        branch branch {.samples=std::move(_samples), .true_branch=false};
        return (void)recursive_greedy_splitting(state, null_split, branch);
    }

    // Constructor.
    // The learning algorithm is initialized with a dataset, a configuration/grouping of the
    // attributes and features, then some hyperparameter determining the structure of the decision-tree.
    dt_learning::dt_learning(
        const dataset& _dataset, const config& _config, int _max_features, int _min_samples_leaf, int _min_samples_split
    )
        :   dataset_(_dataset)
        ,   config_(_config)
        ,   max_features_(_max_features)
        ,   min_samples_leaf_(_min_samples_leaf)
        ,   min_samples_split_(_min_samples_split)
    {}


} // namespace PhantomX