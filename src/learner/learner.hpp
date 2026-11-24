#include <string>

#include <dataset.hpp>
#include <configuration.hpp>
#include <decision_tree.hpp>

namespace PhantomX {

class learner {

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

    // The maximum number of features to be considered
    // while evaluating the quality of splits.
    int max_features_=-1;
    // The minimum number of samples required to be
    // at any leaf node of the decision tree.
    int min_samples_leaf_=-1;
    // the minimum number of sample required at a
    // node before it is considered eligible for 
    // splitting.
    int min_samples_split_=-1;

public:

    // Trains the model with the specified dataset, with respect to the specified
    // features configuration data. The resulting model would be saved to a binary file of 
    // the specified name.
    result learn(const dataset& _dataset, const config& _config, const std::string& _filename);
};

} // namespace PhantomX