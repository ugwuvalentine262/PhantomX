// ============================================================================
//  boost.cpp
// ============================================================================
//
//  Implements `boost::learner`, declared in boost.hpp: the learner for our
//  custom boosting-based algorithm (this is NOT gradient boosting).
//
//  What is implemented here:
//
//    - the constructor,
//    - every hyper-parameter setter (with its validation rule),
//    - every hyper-parameter getter.
//
// ============================================================================

#include <cstdint>
#include <stdexcept>
#include <numeric>
#include <chrono>

#include <thinker/decision_tree.hpp>

#include "decision_tree.hpp"
#include "boost.hpp"

namespace PhantomX {

// ------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------

// Stores references to the dataset and the feature configuration (neither
// is copied, so both must outlive the learner) and records the goal id.
//
// Every other member keeps the default value given by its in-class
// initializer in boost.hpp:
//
//     max_features_      = -1    (no limit)
//     min_samples_leaf_  = -1    (constraint ignored)
//     min_samples_split_ = -1    (constraint ignored)
//     growth_            = 2.5
//     max_population_    = 20
//     min_accuracy_      = 0.5
//
// `ensemble_` starts out empty; it is filled in by `learn()`.
boost::learner::learner(const dataset& _dataset,
                        const config& _config,
                        std::uint64_t _goal_id)
    : dataset_(_dataset),
      config_(_config),
      goal_id_(_goal_id)
{
}

// ------------------------------------------------------------------------
// Hyper-parameter setters
//
// Each setter returns true iff the argument is accepted. A rejected
// argument leaves the stored value UNCHANGED.
// ------------------------------------------------------------------------

// Accepts any non-zero value. A negative value is accepted but means
// "no limit" (every feature is considered at every split). Only zero is
// rejected, since considering zero features per split is meaningless.
// The rule is inherited from `decision_tree::learner`.
bool boost::learner::set_max_features(int i)
{
    if (i == 0)
        return false;

    max_features_ = i;
    return true;
}

// Accepts any non-zero value. A negative value is accepted but disables
// the constraint. Only zero is rejected, since a leaf must hold at least
// one sample. The rule is inherited from `decision_tree::learner`.
bool boost::learner::set_min_samples_leaf(int i)
{
    if (i == 0)
        return false;

    min_samples_leaf_ = i;
    return true;
}

// Accepts any non-zero value. A negative value is accepted but disables
// the constraint. Only zero is rejected. The rule is inherited from
// `decision_tree::learner`.
bool boost::learner::set_min_samples_split(int i)
{
    if (i == 0)
        return false;

    min_samples_split_ = i;
    return true;
}

// Accepts only values strictly greater than 0 (the constant `k` in
// f(x) = e^(k * x) - 1 must be positive for f to be increasing on
// (0, 1]). Written as a positive comparison so that NaN is rejected too:
// every comparison involving NaN is false.
bool boost::learner::set_growth_rate(float v)
{
    if (!(v > 0.0f))
        return false;

    growth_ = v;
    return true;
}

// Accepts only values strictly greater than 0: an ensemble that may hold
// at most zero (or a negative number of) trees is meaningless.
bool boost::learner::set_max_population(int i)
{
    if (i <= 0)
        return false;

    max_population_ = i;
    return true;
}

// Accepts only values in the half-open range (0, 1]: strictly greater
// than 0 and less than or equal to 1. Written as a positive range check
// so that NaN is rejected too (every comparison involving NaN is false).
bool boost::learner::set_min_accuracy(float v)
{
    if (!(v > 0.0f && v <= 1.0f))
        return false;

    min_accuracy_ = v;
    return true;
}

// ------------------------------------------------------------------------
// Hyper-parameter getters
//
// Each getter simply returns the value currently stored in the matching
// member. They never modify the learner and never fail.
// ------------------------------------------------------------------------

// Returns `max_features_`. A negative value means "no limit".
int boost::learner::get_max_features() const
{
    return max_features_;
}

// Returns `min_samples_leaf_`. A negative value means the constraint is
// ignored.
int boost::learner::get_min_samples_leaf() const
{
    return min_samples_leaf_;
}

// Returns `min_samples_split_`. A negative value means the constraint is
// ignored.
int boost::learner::get_min_samples_split() const
{
    return min_samples_split_;
}

// Returns `growth_`. Always strictly greater than 0, because the setter
// rejects anything else and the default is 2.5.
float boost::learner::get_growth_rate() const
{
    return growth_;
}

// Returns `max_population_`. Always strictly greater than 0, because the
// setter rejects anything else and the default is 20.
int boost::learner::get_max_population() const
{
    return max_population_;
}

// Returns `min_accuracy_`. Always in the range (0, 1], because the setter
// rejects anything else and the default is 0.5.
float boost::learner::get_min_accuracy() const
{
    return min_accuracy_;
}

// ------------------------------------------------------------------------
// Public interface
// ------------------------------------------------------------------------

// Trains the boosting model: grows up to `max_population_` decision trees
// for the goal `goal_id_`, pushes them into `ensemble_` one stage at a
// time, and returns the resulting `boost` object.
boost boost::learner::learn()
{
    // Invalid dataset or configuration?
    if (!static_cast<bool>(dataset_) ||
        !static_cast<bool>(config_)) {
        return {};
    }

    decision_tree::learner::weights weights(
        dataset_.size(), 1.0 / dataset_.size());

    temporary temp;

    indexes samples(dataset_.size());
    std::iota(samples.begin(), samples.end(), 0);

    decision_tree::learner dtl(dataset_, config_, weights);

    // Set hyperparameters.
    dtl.set_growth_rate(get_growth_rate());
    dtl.set_min_samples_split(get_min_samples_split());
    dtl.set_min_samples_leaf(get_min_samples_leaf());
    dtl.set_max_features(get_max_features())

    float accuracy = 0.0;
    int population = 0;

    // Stop training only when we have reached a desirable accuracy,
    // or we have hit the limit of the maximum population.
    while (accuracy < min_accuracy_ &&
        population < max_population_) {

            // Add the trained tree to the ensemble.
            ensemble_.push_back(dtl.learn(samples));

            // Evaluate ensemble's accuracy.
            evaluate(ensemble_);

            ++population;
    }

    return boost {
        .goal_id_ = goal_id_,
        .ensemble_ = std::move(ensemble_)
    };
}

} // namespace PhantomX
