#include <iostream>
#include <chrono>

#include "learner.hpp"
#include "dt_learning.hpp"

namespace PhantomX {

    learner::result learner::train(const std::string& _dataset_file)
    {
        auto start = std::chrono::steady_clock::now();

        if (!dataset_.load(_dataset_file)) {
            std::cerr << "Unable to load dataset: '" << _dataset_file << "'\n";
            return result();
        }

        indices indices(dataset_.size());
        std::iota(indices.begin(), indices.end(), 0);
        indexes samples(std::move(indices));
        dt_learning dtl(dataset_, max_features_, min_samples_leaf_, min_samples_split_);
        decision_tree dt = dtl(samples);
        auto finish = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();
        dataset_.clear();
        return result(dt, dtl.max_depth_, elapsed);
    }

    learner::learner(
            int _max_features
        ,   int _min_samples_leaf
        ,   int _min_samples_split
    )
        :   max_features_(_max_features)
        ,   min_samples_leaf_(_min_samples_leaf)
        ,   min_samples_split_(_min_samples_split)
    {}

} // namespace PhantomX