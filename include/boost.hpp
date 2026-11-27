// ============================================================================
//  boost.hpp
// ============================================================================
//
//  Declares `boost`, the top-level container for a gradient-boosted ensemble
//  of decision trees.
//
//  Role of this type:
//
//    - `boost` is a *model* type, not a trainer or predictor. It holds the
//      trained state (the goal id and the ensemble of decision trees) and
//      knows how to serialize itself.
//
//    - Training is planned for the nested `boost::learner`, and inference is
//      planned for the nested `boost::thinker`. Both are forward declared here;
//      boosting training and prediction are not implemented yet.
//
//    - Because `learner` and `thinker` are nested types of `boost`, they
//      have access to `boost`'s private members (`goal_id_`, `ensemble_`)
//      and can populate / read them directly without extra accessors.
//
// ============================================================================

#ifndef BOOST_HPP
#define BOOST_HPP

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include <decision_tree.hpp>

namespace PhantomX {

// ----------------------------------------------------------------------------
// boost
//
// The model container for a planned gradient-boosting algorithm. Training and
// prediction are not implemented yet; this type currently owns model state and
// serialization only.
//
// Concretely, a `boost` instance represents a single trained gradient-boosted
// model for one goal: it owns the target/goal identifier and the sequence of
// decision trees that, taken together, approximate the target function.
// ----------------------------------------------------------------------------
struct boost {

    // ------------------------------------------------------------------------
    // Nested algorithm types (forward declarations)
    // ------------------------------------------------------------------------

    // The gradient boosting learning algorithm. Defined in its own header.
    // Fills in `ensemble_` and `goal_id_` when training.
    class learner;

    // The gradient boosting prediction algorithm. Defined in its own header.
    // Reads `ensemble_` and `goal_id_` when predicting.
    class thinker;

    // ------------------------------------------------------------------------
    // Public type aliases
    // ------------------------------------------------------------------------

    // The collection of decision-tree models that make up the gradient
    // boosting ensemble. One entry per boosting stage.
    using ensemble = std::vector<decision_tree>;

private:

    // ------------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------------

    // The goal/target id of the boosting model. Identifies which goal this
    // ensemble was trained for. Kept in sync with
    // `boost::learner::goal_id_`.
    std::uint64_t goal_id_;

    // The decision-tree models that make up the gradient boosting model,
    // ordered by boosting stage. Populated by `boost::learner` and consumed
    // by `boost::thinker`.
    ensemble ensemble_;

public:

    // ------------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------------

    // Writes the nodes and leaf data of the gradient boosting model
    // to the specified output-stream. Returns true iff the
    // operation completed successfully.
    //
    // The goal id and every tree in the ensemble are serialized; each tree
    // delegates to `decision_tree::write()`.
    bool write(std::ostream& _os) const;

    // Reads the nodes and leaf data of the gradient boosting model
    // from the specified input-stream. Returns true iff the
    // operation completed successfully.
    //
    // Reconstructs the goal id and every tree in the ensemble from the
    // stream; each tree delegates to `decision_tree::read()`.
    bool read(std::istream& _is);
};

} // namespace PhantomX

#endif // BOOST_HPP
