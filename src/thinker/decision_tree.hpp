// ============================================================================
//  decision_tree thinker
// ============================================================================
//
//  Declares the prediction-side interface for a trained `decision_tree`.
//  Training and model serialization live elsewhere; this type only keeps
//  references to an existing model and its feature configuration and exposes
//  the prediction operation.
//
//  Current implementation status
//  ------------------------------
//  Tree traversal and leaf-to-label expansion are implemented in
//  `decision_tree.cpp`. Invalid or incomplete model data produces an empty
//  result. Callers that make repeated predictions can provide a `temporary`
//  instance to reuse the traversal buffers between calls.
//
//  Ownership and lifetime
//  ----------------------
//  A thinker does not own either object supplied to its constructor. The
//  referenced `decision_tree` and `config` must outlive the thinker. Both are
//  stored as const references because prediction must not mutate the model or
//  the feature grouping configuration.
//
//  Prediction result
//  -----------------
//  A result maps a canonical `label` to its double-precision prediction
//  weight. `std::map` keeps labels ordered. Each generated label is assigned
//  the exact weight of the leaf that produced it.
//
// ============================================================================

#ifndef DECISION_TREE_THINKER_HPP
#define DECISION_TREE_THINKER_HPP

#include <cstdint>
#include <map>

#include <decision_tree.hpp>

namespace PhantomX {

// ----------------------------------------------------------------------------
// decision_tree::thinker
// ----------------------------------------------------------------------------
//
// Prediction is deliberately represented as a nested type of `decision_tree`.
// This mirrors the learner/model relationship and allows the implementation to
// access the model's private node and leaf storage when traversal is added.
//
struct decision_tree::thinker {

    // Public aliases used by the prediction API. Keeping these names close to
    // the thinker makes it clear which type represents one input sample and
    // which type represents a completed prediction.
    using input = indexes;

    // ------------------------------------------------------------------------
    // label
    // ------------------------------------------------------------------------
    //
    // Canonical key for a predicted label combination. Each populated slot is
    // intended to hold the FNV-1a hash of a key from `config::y`; unused slots
    // contain UINT64_MAX as a sentinel value.
    //
    // The fixed eight-slot representation matches the maximum number of
    // configured argument groups supported by the dataset/configuration layer.
    // Labels are compared lexicographically so they can be used as keys in a
    // `std::map`.
    struct label {

        // Hashes for the configured label arguments. Slots beyond the active
        // configuration argument count retain the sentinel value.
        std::uint64_t args[8] = {
            0xffffffffffffffffull,
            0xffffffffffffffffull,
            0xffffffffffffffffull,
            0xffffffffffffffffull,
            0xffffffffffffffffull,
            0xffffffffffffffffull,
            0xffffffffffffffffull,
            0xffffffffffffffffull
        };

        // Provides strict lexicographical ordering over all eight slots.
        // Equal labels compare false in both directions.
        bool operator<(const label& other) const;
    };

    // ------------------------------------------------------------------------
    // result
    // ------------------------------------------------------------------------
    //
    // Stores weighted predictions. A generated label receives the exact leaf
    // weight supplied to `add_prediction()`; the current model is expected not
    // to generate duplicate label combinations.
    //
    // Prediction returns an empty result only when traversal or leaf expansion
    // encounters invalid or incomplete model/configuration data.
    class result {

        // Ordered mapping from canonical label to accumulated weight.
        std::map<label, double> predictions_;

    public:

        // Assigns `weight` to `lbl`. A missing label is created by
        // `std::map::operator[]`; an existing entry is replaced so each label
        // retains exactly the weight supplied by its generating leaf.
        void add_prediction(const label& lbl, double weight);

        // Returns the complete ordered prediction map. The reference remains
        // valid until this result is modified or destroyed.
        const std::map<label, double>& predictions() const;

        // Removes all predicted labels and weights. The result remains valid
        // and reusable after this call.
        void clear();

        // Adds every label/weight pair in `other` to this result. This is the
        // explicit accumulation operation for combining separate prediction
        // results; individual leaf-generated labels use `add_prediction()` and
        // receive their leaf weight exactly.
        result& operator+=(const result& other);
    };

    using output = result;

    // ------------------------------------------------------------------------
    // temporary
    // ------------------------------------------------------------------------
    //
    // Reusable scratch storage for one prediction. These buffers are cleared
    // before each core prediction, but their capacities are retained so
    // repeated calls do not repeatedly allocate memory for traversal paths,
    // visited-node flags, intersections, or matching hash rows.
    //
    // A temporary instance must not be used concurrently by multiple
    // predictions. It is mutable caller-owned workspace, not model state.
    struct temporary {
        // Marks nodes visited during the current traversal. Its size is made
        // equal to the model's node count before traversal begins.
        std::vector<bool> visited;

        // Decision-path rows. Row zero is the input sample; left-hand splits
        // append progressively narrower intersections.
        std::vector<indexes> path;

        // For each configured argument row, stores the compatible y-hashes
        // found at the selected leaf.
        std::vector<std::vector<std::uint64_t>> data;

        // Scratch result for an attribute-set intersection while matching a
        // leaf argument row.
        indexes intersection;

        // The next path row produced while evaluating an internal split.
        indexes next_path;
    };

private:

    // ------------------------------------------------------------------------
    // Referenced prediction inputs
    // ------------------------------------------------------------------------

    // Trained tree being traversed. The thinker does not own this object.
    const decision_tree& model_;

    // Feature/target grouping used to interpret node split indexes and leaf
    // coordinates. The thinker does not own this object.
    const config& config_;

public:

    // ------------------------------------------------------------------------
    // Construction and prediction
    // ------------------------------------------------------------------------

    // Binds the thinker to an existing trained model and its configuration.
    // Both references must remain valid for the thinker's entire lifetime.
    thinker(const decision_tree& _model, const config& _config);

    // Predicts labels for a sample represented by its sorted attribute indexes.
    // This convenience overload creates per-call scratch storage and delegates
    // to the reusable-workspace overload below.
    //
    // If a path, node, leaf, or configured argument row is invalid, the method
    // returns an empty result rather than exposing a partial label.
    output predict(const input& _sample) const;

    // Runs the core prediction algorithm using caller-provided reusable
    // temporary storage. Active scratch contents are reset before use, while
    // allocated capacity—including retained path-row capacity—is kept for
    // subsequent calls.
    output predict(const input& _sample, temporary& _temporary) const;
};

} // namespace PhantomX

#endif // DECISION_TREE_THINKER_HPP
