#ifdef DECISION_TREE_THINKER_HPP
#define DECISION_TREE_THINKER_HPP

#include <decision_tree.hpp>

namespace PhantomX {

    // The decision-tree thinker is responsible for making predictions using a trained decision-tree model.
    struct decision_tree::thinker {
        struct result {
            coordinates coordinates; // the predicted coordinates for the given sample.
        };

        // Constructor initializes the thinker with a trained decision-tree model
        // and the configuration/grouping of the attributes and features.
        thinker(const decision_tree& _model, const config& _config);
        // Predicts the target value for the given sample using the trained decision-tree model.
        coordinates predict(const indexes& _sample) const;
    private:
        const decision_tree& model_; // reference to the trained decision-tree model.
        const config& config_; // reference to the configuration/grouping of the attributes and features.
    };

} // namespace PhantomX
#endif // DECISION_TREE_THINKER_HPP