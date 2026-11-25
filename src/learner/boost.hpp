#ifndef BOOST_LEANER_HPP
#define BOOST_LEARNER_HPP

#include <string>

#include <boost.hpp>
#include <dataset.hpp>
#include <configuration.hpp>
#include <decision_tree.hpp>

namespace PhantomX {

// This class contains our implementation of a gradient boosting learning algorithm.
class boost::learner {

public:

    struct result {
        int success=0;
        int elapsed=0;
        int nodes=0;
        int leafs=0;
    public:
        result(int _max_depth, int _elapsed);
        result()=default;
    };

private:

    // -------- Hyper-parameters ------- //

    // The number of features to be considered
    // while evaluating the quality of splits.
    int max_features_=-1;
    // The minimum number of samples required to be
    // at any leaf node of the decision tree.
    int min_samples_leaf_=-1;
    // the minimum number of sample required at a
    // node before it is considered eligible for 
    // splitting.
    int min_samples_split_=-1;
    // exponential growth-rate.
    float growth=2.5;

public:

    bool set_max_features(int i);
    bool set_min_samples_leaf(int i);
    bool set_min_samples_split(int i);
    bool set_growth_rate(float v);

    // Trains the model with the specified dataset, with respect to the specified
    // features configuration data.
    result learn(const dataset& _dataset, const config& _config, const std::string& _filename);
};

} // namespace PhantomX

#endif // BOOST_LEARNER_HPP