#include <iostream>
#include <chrono>
#include <numeric>

#include "learner.hpp"
#include "dt_learning.hpp"

namespace PhantomX {

    learner::result learner::learn(const dataset& _dataset, const config& _config, const std::string& _filename)
    {
        auto start = std::chrono::steady_clock::now();

        indexes samples(_dataset.size());
        std::iota(samples.begin(), samples.end(), 0);
        dt_learning dtl(_dataset, _config, max_features_, min_samples_leaf_, min_samples_split_);
        decision_tree dt = dtl(std::move(samples));
        auto finish = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();
        return result(dt, dtl.max_depth_, elapsed);
    }

} // namespace PhantomX