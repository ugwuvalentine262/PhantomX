// ============================================================================
//  Declares the planned `boost::learner` interface for training an ensemble
//  of decision trees using our custom boosting-based algorithm (this is NOT
//  gradient boosting). The implementation is not yet complete.
// ============================================================================
//
//  Planned design:
//
//    1. A `boost::learner` owns the boosting hyper-parameters (the same
//       tree-level knobs exposed by `decision_tree::learner`, plus a
//       growth rate, maximum population and minimum accuracy) and exposes a
//       single `learn()` entry point.
//
//    2. `learn()` returns a fully trained `boost` model. Internally it
//       populates that model's `ensemble_` one stage at a time, growing at
//       most `max_population_` trees for the goal identified by `goal_id_`.
//
//    3. Each individual tree is produced by `decision_tree::learner::learn()`,
//       which already accepts a subset of sample indexes — so a boosting
//       round can reuse it directly.
//
// ============================================================================

#ifndef BOOST_LEARNER_HPP
#define BOOST_LEARNER_HPP

#include <cstdint>
#include <string>

#include <boost.hpp>           // declares `boost` (goal_id_, ensemble_, ...)
#include <dataset.hpp>         // declares `dataset`, `indexes`, `coordinates`
#include <configuration.hpp>   // declares `config` (feature grouping data)
#include <decision_tree.hpp>   // declares `decision_tree` and its learner

namespace PhantomX {

// ----------------------------------------------------------------------------
// boost::learner
//
// Our implementation of a custom boosting-based learning algorithm (not
// gradient boosting). Declared as a nested type of `boost` so it can populate
// the model's `ensemble_` vector directly.
// ----------------------------------------------------------------------------
class boost::learner {

    // -------- Inputs ------- //

    // Read-only reference to the dataset used to train the model.
    // The dataset must outlive the learner.
    const dataset& dataset_;

    // Feature grouping/configuration data (read-only reference).
    // The config must outlive the learner.
    const config& config_;

    // Mirrors `boost::goal_id_`. Identifies which goal this learner is
    // boosting for.
    const uint64_t goal_id_;

    // -------- Outputs ------- //

    // The ensemble of decision trees produced by the learning algorithm,
    // one tree per boosting stage. Mirrors `boost::ensemble_`.
    boost::ensemble ensemble_;

    // -------- Hyper-parameters ------- //

    // The number of features to be considered
    // while evaluating the quality of splits.
    //
    // Mirrors `decision_tree::learner::max_features_`. A NEGATIVE value
    // means "no limit": every feature is considered at every split.
    // Default: -1 (disabled).
    int max_features_ = -1;

    // The minimum number of samples required to be
    // at any leaf node of the decision tree.
    //
    // Mirrors `decision_tree::learner::min_samples_leaf_`. A NEGATIVE value
    // means the constraint is ignored. Default: -1 (disabled).
    int min_samples_leaf_ = -1;

    // The minimum number of samples required at a
    // node before it is considered eligible for
    // splitting.
    //
    // Mirrors `decision_tree::learner::min_samples_split_`. A NEGATIVE value
    // means the constraint is ignored. Default: -1 (disabled).
    int min_samples_split_ = -1;

    // Exponential growth-rate.
    //
    // Same role as `decision_tree::learner::growth_`: the constant `k` in
    //     f(x) = e^(k * x) - 1,     x in (0, 1]
    // used to weight leaf nodes by purity. Must be strictly positive
    // (enforced by `set_growth_rate()`). Default: 2.5.
    //
    // NOTE: this is NOT a boosting shrinkage / learning-rate.
    float growth_ = 2.5;

    // The maximum number of boosting stages to run, i.e. the upper bound on
    // the number of trees grown in the ensemble. Must always be strictly
    // positive (enforced by `set_max_population()`). Default: 20.
    int max_population_ = 20;

    // The minimum accuracy, expressed as a fraction in the range (0, 1].
    // Must always lie in that range (enforced by `set_min_accuracy()`).
    // Default: 0.5.
    float min_accuracy_ = 0.5;

public:

    // ------------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------------

    // Initializes the learner with a dataset, a feature configuration, and
    // the goal id.
    //
    // `_dataset` and `_config` are stored by reference and must outlive the
    // learner.
    learner(const dataset& _dataset,
            const config& _config,
            uint64_t _goal_id);

    // ------------------------------------------------------------------------
    // Hyper-parameter setters
    //
    // Each returns true iff the argument is accepted, false if rejected.
    // The acceptance rules are inherited from `decision_tree::learner`.
    // ------------------------------------------------------------------------

    // Sets `max_features_`.
    // Returns true iff `i` is non-zero (a negative value returns true but
    // disables the constraint).
    bool set_max_features(int i);

    // Sets `min_samples_leaf_`.
    // Returns true iff `i` is non-zero (a negative value returns true but
    // disables the constraint).
    bool set_min_samples_leaf(int i);

    // Sets `min_samples_split_`.
    // Returns true iff `i` is non-zero (a negative value returns true but
    // disables the constraint).
    bool set_min_samples_split(int i);

    // Sets `growth_`, the exponential growth constant `k` in
    // f(x) = e^(k * x) - 1 used to weight leaf nodes by purity.
    // Returns true iff `v` is strictly greater than 0, false otherwise.
    bool set_growth_rate(float v);

    // Sets `max_population_`, the maximum number of boosting stages / trees
    // in the ensemble.
    // Returns true iff `i` is strictly greater than 0, false otherwise.
    bool set_max_population(int i);

    // Sets `min_accuracy_`.
    // Returns true iff `v` is in the range (0, 1] (strictly greater than 0
    // and less than or equal to 1), false otherwise.
    bool set_min_accuracy(float v);

    // ------------------------------------------------------------------------
    // Hyper-parameter getters
    //
    // Each getter is the counterpart of the setter of the same name and
    // returns the value currently stored in the corresponding member. Getters
    // never modify the learner and never fail.
    // ------------------------------------------------------------------------

    // Returns `max_features_`, the number of features considered while
    // evaluating the quality of splits. A NEGATIVE value means "no limit":
    // every feature is considered at every split. Default: -1 (disabled).
    int get_max_features() const;

    // Returns `min_samples_leaf_`, the minimum number of samples required at
    // any leaf node. A NEGATIVE value means the constraint is ignored.
    // Default: -1 (disabled).
    int get_min_samples_leaf() const;

    // Returns `min_samples_split_`, the minimum number of samples required at
    // a node before it is eligible for splitting. A NEGATIVE value means the
    // constraint is ignored. Default: -1 (disabled).
    int get_min_samples_split() const;

    // Returns `growth_`, the exponential growth constant `k` in
    // f(x) = e^(k * x) - 1 used to weight leaf nodes by purity. The returned
    // value is always strictly greater than 0. Default: 2.5.
    float get_growth_rate() const;

    // Returns `max_population_`, the maximum number of boosting stages /
    // trees in the ensemble. The returned value is always strictly greater
    // than 0. Default: 20.
    int get_max_population() const;

    // Returns `min_accuracy_`. The returned value is always in the range
    // (0, 1]. Default: 0.5.
    float get_min_accuracy() const;

    // ------------------------------------------------------------------------
    // Public interface
    // ------------------------------------------------------------------------

    // Trains the boosting model and returns the resulting `boost` object
    // (whose `ensemble_` vector holds one decision tree per boosting stage).
    // The dataset, feature configuration, and goal id are taken from the
    // members initialized by the constructor.
    boost learn();
};

} // namespace PhantomX

#endif // BOOST_LEARNER_HPP
