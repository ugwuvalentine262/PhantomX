// ============================================================================
//  boost thinker implementation
// ============================================================================
//
//  Each boosted model contains an ordered ensemble of decision trees. The
//  boost thinker sends one sample to every tree thinker, reuses one traversal
//  workspace, and combines the resulting label weights with
//  `result::operator+=`.
//
// ============================================================================

#include "boost.hpp"

namespace PhantomX {

// ----------------------------------------------------------------------------
// boost::thinker::thinker
// ----------------------------------------------------------------------------
//
// Binds the thinker to the existing ensemble and configuration. The references
// are intentionally non-owning; the caller must keep both inputs alive. The
// constructor performs no model validation because the decision-tree thinker
// handles structural and traversal validation when each tree is predicted.
boost::thinker::thinker(const boost& _model, const config& _config)
    : model_(_model),
      config_(_config) {}

// ----------------------------------------------------------------------------
// boost::thinker::predict(input)
// ----------------------------------------------------------------------------
//
// Convenience overload. Scratch storage is local to this prediction and is
// reused across all trees by the delegated overload. The returned output owns
// its prediction map after this function returns; the temporary is only
// traversal workspace.
boost::thinker::output
boost::thinker::predict(const input& _sample) const {
    temporary scratch;
    return predict(_sample, scratch);
}

// ----------------------------------------------------------------------------
// boost::thinker::predict(input, temporary&)
// ----------------------------------------------------------------------------
//
// Core ensemble prediction. Each tree receives the same input sample and
// shared configuration. The decision-tree thinker resets the reusable
// temporary before traversing that tree, so no tree sees stale path, visited,
// intersection, or leaf-matching state from the previous tree.
//
// A tree assigns its leaf weight to each generated label. This method combines
// those independent tree results with `operator+=`, so a label's final ensemble
// weight is the sum of its contributions from all trees.
boost::thinker::output
boost::thinker::predict(const input& _sample,
                        temporary& _temporary) const {
    output ensemble_output;

    // Preserve ensemble order. Although the current result type is a map and
    // therefore does not expose ordering, keeping this loop ordered preserves
    // the model's defined stage sequence and makes future stage-sensitive
    // behavior possible.
    for (const decision_tree& tree : model_.ensemble_) {
        // The tree thinker is a lightweight adapter around one tree. It keeps
        // a reference to `tree` and reuses the caller-provided workspace.
        decision_tree::thinker tree_thinker(tree, config_);
        ensemble_output += tree_thinker.predict(_sample, _temporary);
    }

    return ensemble_output;
}

} // namespace PhantomX
