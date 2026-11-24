#include <algorithm>
#include <random>
#include <numeric>

#include "decision_tree.hpp"
#include "objective.hpp"

namespace PhantomX {

    // The recursive greedy splitting algorithm is the core of the decision tree learning algorithm.
    // This is a process used by the decision tree to divide a dataset into smaller subsets based on the 
    // features that best separate them into meaningful clusters.
    int decision_tree::learner::recursive_greedy_splitting(state& _state, const split& _split, const branch& _branch) {
        int node_idx = node_idx_++;
        split best_split;
        _state.apply(config_, _split, _branch);

        for (std::size_t g=0; g<config_.x.size(); g++) {
            for (std::size_t i=0; i<=_state.paths[0].size(); i++) {
                split split = divide(_branch.samples, _state, g, i);
                if (split.score>best_split.score) {
                    best_split = std::move(split);
                }
            }
        }
        // did we find a good split?
        if (best_split.score>_branch.score) {
            add_node(   node_idx
                    ,   recursive_greedy_splitting(_state, best_split, best_split.left)
                    ,   recursive_greedy_splitting(_state, best_split, best_split.right)
                    ,   best_split
                );
        } else { // no good split was found.
            add_leaf(node_idx, _branch);
        }
        _state.undo(_split, _branch);
        return node_idx;
    }

    // Adds a node to the decision-tree model.
    void decision_tree::learner::add_node(int _node_idx, int _left_node, int _right_node, const split& _split) {
        if (static_cast<std::size_t>(_node_idx)>=nodes_.size()) {
            nodes_.resize(_node_idx + 1);
        }

        auto& node = nodes_[_node_idx];

        node.left=_left_node;
        node.right=_right_node;
        node.group=_split.group;
        node.index=_split.index;
    }

    // Adds a leaf node to the decision-tree model.
    void decision_tree::learner::add_leaf(int _node_idx, const branch& _branch) {
        if (static_cast<std::size_t>(_node_idx)>=nodes_.size()) {
            nodes_.resize(_node_idx + 1);
        }

        auto& node = nodes_[_node_idx];

        node.left=0;
        node.right=0;
        node.group=0;
        node.index=leafs_.size();

        leafs_.emplace_back(decision_tree::leaf{.args=_branch.args, .weight=0});
    }

    // Updates the current state with respect to the given parameters.
    void decision_tree::learner::state::apply(const config& _config, const decision_tree::learner::split& _split, const branch& _branch)
    {
        if (!_branch.true_branch) { return; }
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
    void decision_tree::learner::state::undo(const decision_tree::learner::split& _split, const branch& _branch) {
        if (!_branch.true_branch) { return; }
        for (size_t i=0; i < _branch.samples.size(); i++) {
            paths[_branch.samples[i]].pop_back();
        }
    }

    // Constructor initializes the state with a dataset.
    decision_tree::learner::state::state(const dataset& _dataset) {
        paths.reserve(_dataset.size());
        for (auto& sample : _dataset.samples) {
            paths.emplace_back(decision_path{group(sample.x)});
        }
    }

    // Constructor.
    // The new group is formed from the intersection of the specified
    // groups. The newly created group is initially a leaf.
    decision_tree::learner::state::group::group(const config::group& _a, decision_tree::learner::state::group& _b)
        :   group(_a.indexes & _b.indices)
    {
        _b.leaf=false;
    }

    // Constructor.
    // Initializes the group with indexes.
    // The newly created group is initially a leaf.
    decision_tree::learner::state::group::group(const indexes& _indexes)
        :   indices(_indexes)
        ,   leaf(true)
    {}

    // Splits the samples using the specified feature/group, and returns the resulting split data.
    decision_tree::learner::split decision_tree::learner::divide(const indexes& _samples, const state& _state, int _group, int _index) const {
        split split;

        // number of samples required for splitting at the node
        // must meet up to the minimum requirement.
        if (min_samples_split_>0
            &&static_cast<std::size_t>(min_samples_split_)>_samples.size()) {
            return split;
        }

        auto& first = _state.paths.at(_samples.front());

        int cnt = std::count_if(first.begin(), first.end(), [](const state::group& g)
                        {
                            return g.leaf;
                        });

        int llc=cnt, rlc=cnt;
        int k=config_.args.size();

        first[_index].leaf?rlc:++llc;

        Objective::Matrix left(llc, std::vector<indexes>(k));
        Objective::Matrix right(rlc, std::vector<indexes>(k));

        for (auto sample : _samples) {
            auto& path = _state.paths[sample];
            auto intersect = path[_index].indices & config_.x[_group].indexes;
            for (std::size_t i=0,j=0; i < path.size(); i++) {
                if (!path[i].leaf) {
                    continue;
                }

                if (intersect.empty()) {
                    for (auto h=0; h < k; h++) {
                        split.right.samples.push_back(sample);
                        right[j][h].push_back(sample);
                    }
                } else {
                    for (auto h=0; h < k; h++) {
                        split.left.samples.push_back(sample);
                        left[j][h].push_back(sample);
                    }
                }
                ++j;
            }
        }

        // number of samples required at the leaf node
        // must meet up to the minimum requirement.
        if (min_samples_leaf_>0) {
            if (static_cast<std::size_t>(min_samples_leaf_)>split.right.samples.size()
                ||static_cast<std::size_t>(min_samples_leaf_)>split.left.samples.size()) {
                return split;
            }
        }

        auto r_res = Objective::solve(right, rlc);
        auto l_res = Objective::solve(left, llc);

        split.right.score=r_res.intersectionSize;
        split.right.args=r_res.selectedIndices;
        split.left.score=l_res.intersectionSize;
        split.left.args=l_res.selectedIndices;
        split.score=split.left.score+split.right.score;
        split.index=_index;
        split.group=_group;

        return split;
    }

    // Calls the main learning algorithm for the decision-tree model.
    void decision_tree::learner::operator()(indexes _samples) {
        split null_split;
        state state(dataset_);
        branch branch {.samples=std::move(_samples), .true_branch=false};
        return (void)recursive_greedy_splitting(state, null_split, branch);
    }

    // Constructor.
    // The learning algorithm is initialized with a dataset, a configuration/grouping of the
    // attributes and features, then some hyperparameter determining the structure of the decision-tree.
    decision_tree::learner::learner(
        const dataset& _dataset, const config& _config, int _max_features, int _min_samples_leaf, int _min_samples_split
    )
        :   dataset_(_dataset)
        ,   config_(_config)
        ,   max_features_(_max_features)
        ,   min_samples_leaf_(_min_samples_leaf)
        ,   min_samples_split_(_min_samples_split)
    {}

} // namespace PhantomX