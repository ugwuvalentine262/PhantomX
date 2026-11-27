// ============================================================================
//  boost thinker
// ============================================================================
//
//  Declares the prediction-side interface for a boosted ensemble. A boost model
//  is an ordered collection of decision trees; prediction delegates traversal
//  of each individual tree to `decision_tree::thinker` and accumulates the
//  returned label weights in ensemble order.
//
//  The boost thinker deliberately reuses the decision-tree thinker's input,
//  output, and temporary types. This keeps result semantics consistent between
//  one tree and an ensemble. One scratch workspace can be reused for every tree
//  in one prediction call, avoiding a separate traversal-buffer allocation per
//  ensemble member.
//
// ============================================================================

#ifndef BOOST_THINKER_HPP
#define BOOST_THINKER_HPP

#include <boost.hpp>

#include "decision_tree.hpp"

namespace PhantomX {

// ----------------------------------------------------------------------------
// boost::thinker
// ----------------------------------------------------------------------------
//
// A boost thinker does not own the model or configuration. Both must outlive
// the thinker and remain unchanged while predictions are being made. The model
// contains the ordered ensemble, while the configuration supplies the feature
// and target-group metadata needed by each decision-tree thinker.
struct boost::thinker {

    // ------------------------------------------------------------------------
    // Public API aliases
    // ------------------------------------------------------------------------
    //
    // These aliases intentionally match `decision_tree::thinker`. A caller can
    // therefore use the same input and output vocabulary for one tree and for
    // an ensemble. `temporary` is also the exact same reusable workspace type.
    using input = indexes;
    using result = decision_tree::thinker::result;
    using output = result;
    using temporary = decision_tree::thinker::temporary;

    // ------------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------------

    // Binds the thinker to a trained ensemble and its feature configuration.
    // Neither argument is copied or owned. Both references must remain valid
    // for the entire lifetime of this thinker and during every prediction.
    thinker(const boost& _model, const config& _config);

    // ------------------------------------------------------------------------
    // Prediction
    // ------------------------------------------------------------------------

    // Predicts with a private per-call scratch workspace. This is the simplest
    // overload for callers that do not need to control allocation reuse.
    // Internally it delegates to the overload accepting explicit workspace.
    output predict(const input& _sample) const;

    // Predicts with caller-provided reusable scratch storage. The workspace is
    // cleared and reused by each delegated tree prediction, so the same
    // temporary serves every tree in the ensemble. A temporary must not be
    // shared concurrently by multiple predictions.
    output predict(const input& _sample, temporary& _temporary) const;

private:

    // Non-owning reference to the boosted model. Its ensemble is traversed in
    // the stored order, and every tree contributes to the returned output.
    const boost& model_;

    // Non-owning reference to the configuration shared by all trees in this
    // model. It must describe the same feature/target layout used during
    // training.
    const config& config_;
};

} // namespace PhantomX

#endif // BOOST_THINKER_HPP
