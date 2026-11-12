#include <algorithm>
#include <random>
#include <numeric>

#include "dt_learning.hpp"
#include "objective.hpp"

namespace PhantomX {

    int dt_learning::recursive_greedy_splitting(state& _state, const split& _split, const indexes& _samples, int _depth)
    {
        decision_tree::node node;
        decision_tree::leaf leaf;

        int node_idx = num_of_nodes_++;

        split best_split;

        _state.apply(dataset_, _split, _samples);

        for (const group& g : dataset_.groups_)
        {
            for (int j=0; j < _depth + 1; j++)
            {
                split split = divide(_samples, _state, g, j);

                if (valid(split)) {
                    if (!best_split || (split > best_split)) {
                        best_split = std::move(split);
                    }
                }
            }
        }

        if (best_split) {

            add_node(   node_idx_
                    ,   recursive_greedy_splitting(
                                _state
                            ,   best_split
                            ,   best_split.left_
                            ,   _depth + 1
                        )
                    ,   recursive_greedy_splitting(
                                _state
                            ,   best_split
                            ,   best_split.right_
                            ,   _depth + 1
                        )
                    ,   best_split.index_
                    ,   best_split.group_
                );
        }
        else {
            ++num_of_leafs_;
            max_depth_ = std::max
                    (max_depth_, _depth);

            add_leaf(node_idx_);
        }

        _state.undo(_split, _samples);

        return node_idx;
    }

    dt_learning::split::operator bool() const
    {
        return score_<0?true:false;
    }

    bool dt_learning::split::operator>(const split& _split) const
    {
        return this->score_ > _split.score_;
    }

    void dt_learning::state::apply(const dataset& _dataset, const dt_learning::split& _split, const indexes& _samples)
    {
        if (!_split) { return; }

        for (size_t i=0; i < _samples.size(); i++)
        {
            int index = _samples[i];
            groups& grouping = online_grouping_.at(index);

            auto& operand1 = _dataset.groups_.at(_split.group_);
            auto& operand2 = grouping.at(_split.index_);

            grouping.emplace_back(operand1 & operand2);
        }
    }

    void dt_learning::state::undo(const dt_learning::split& _split, const indexes& _samples)
    {
        if (!_split) { return; }

        for (size_t i=0; i < _samples.size(); i++) {
            online_grouping_[_samples[i]].pop_back();
        }
    }

    dt_learning::split dt_learning::divide(const indexes& _samples, const state& _state, const group& _group, int _depth) const
    {
        split split;

        return split;
    }

    bool dt_learning::valid(const split& _split) const
    {
        return true;
    }

} // namespace PhantomX