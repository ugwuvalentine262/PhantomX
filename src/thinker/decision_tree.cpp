// ============================================================================
//  decision_tree thinker implementation
// ============================================================================
//
//  This translation unit contains the out-of-line member definitions for the
//  nested `decision_tree::thinker` type declared in `decision_tree.hpp`.
//
//  Prediction flow
//  ---------------
//  Prediction starts with the sample's attribute indexes as path row zero.
//  Internal nodes intersect the selected path row with their configured feature
//  group. A non-empty intersection follows the left child and appends the new
//  path row; an empty intersection follows the right child without extending
//  the path. At a leaf, compatible hashes are collected for every configured
//  argument row and expanded into label combinations.
//
// ============================================================================

#include "decision_tree.hpp"

namespace PhantomX {

// ----------------------------------------------------------------------------
// decision_tree::thinker::label::operator<
// ----------------------------------------------------------------------------
//
// `label` is used as the key of `result::predictions_`, which requires a strict
// weak ordering. The label is a fixed-size array rather than a variable-length
// container so that all configured argument combinations have one canonical
// representation, including unused slots containing the sentinel value.
//
// Lexicographical comparison makes the ordering deterministic: the first slot
// that differs decides the result, and completely equal labels compare false.
// The latter is important because an object must not compare less than itself.
bool decision_tree::thinker::label::operator<(
    const decision_tree::thinker::label& other) const {
    for (int i = 0; i < 8; ++i) {
        if (args[i] < other.args[i]) return true;
        if (args[i] > other.args[i]) return false;
    }
    return false;
}

// ----------------------------------------------------------------------------
// decision_tree::thinker::result::add_prediction
// ----------------------------------------------------------------------------
//
// Assigns one leaf weight to a label. Using `operator[]` creates a missing map
// entry, while assignment (rather than addition) guarantees that a generated
// label receives exactly the leaf weight requested by the prediction contract.
void decision_tree::thinker::result::add_prediction(
    const decision_tree::thinker::label& lbl,
    double weight) {
    predictions_[lbl] = weight;
}

// ----------------------------------------------------------------------------
// decision_tree::thinker::result::predictions
// ----------------------------------------------------------------------------
//
// Exposes the accumulated predictions without copying the map. The result
// object remains the owner of the map, so the returned reference is valid only
// while the result remains alive and unmodified in ways that invalidate map
// references.
const std::map<decision_tree::thinker::label, double>&
decision_tree::thinker::result::predictions() const {
    return predictions_;
}

// Removes all stored predictions while leaving the result object reusable.
// `std::map::clear()` may retain implementation-specific allocation resources
// for the map, avoiding unnecessary churn when the result is reused.
void decision_tree::thinker::result::clear() {
    predictions_.clear();
}

// Combines another result into this one. Unlike add_prediction(), which
// assigns the exact weight produced by one leaf, this operator is intended for
// callers combining independently produced prediction results.
decision_tree::thinker::result&
decision_tree::thinker::result::operator+=(const result& other) {
    for (const auto& [label, weight] : other.predictions_) {
        predictions_[label] += weight;
    }
    return *this;
}

// ----------------------------------------------------------------------------
// decision_tree::thinker::thinker
// ----------------------------------------------------------------------------
//
// Binds a thinker to an existing model and configuration. Both inputs are
// stored as references rather than copied: models and configurations are shared
// read-only data, and a thinker must not outlive either referenced object.
decision_tree::thinker::thinker(const decision_tree& _model,
                                const config& _config)
    : model_(_model),
      config_(_config) {}

// ----------------------------------------------------------------------------
// decision_tree::thinker::predict
// ----------------------------------------------------------------------------
//
// The prediction flow is:
//
//   1. Start at the root with the input sample as path row zero.
//   2. For each internal node, intersect the selected path row with the
//      configured feature group. A non-empty intersection follows the left
//      child and appends a new path row; an empty intersection follows the
//      right child without changing the path.
//   3. At a leaf, match each configured argument row against the path row
//      referenced by the corresponding leaf coordinate.
//   4. Expand the matching hashes as a Cartesian product. Each complete label
//      receives exactly the leaf's weight.
//
// Malformed in-memory models and incomplete leaf data return an empty result.
decision_tree::thinker::output
decision_tree::thinker::predict(const input& _sample) const {
    temporary scratch;
    return predict(_sample, scratch);
}

// Core prediction overload using caller-provided reusable scratch storage.
decision_tree::thinker::output
decision_tree::thinker::predict(const input& _sample,
                                temporary& _temporary) const {
    result output;

    // Reset every active per-call buffer while retaining its capacity for
    // reuse. Path rows are handled separately below so their inner index
    // allocations survive across calls.
    _temporary.visited.clear();
    _temporary.visited.resize(model_.nodes_.size(), false);
    _temporary.data.resize(config_.args.size());
    // Resize only after retaining the outer vector. Existing row vectors keep
    // their capacity across calls, and clearing them does not release that
    // capacity.
    for (auto& row : _temporary.data) row.clear();
    _temporary.intersection.clear();
    _temporary.next_path.clear();

    // The initial path row is the complete set of attributes supplied by the
    // caller. Keep previously allocated path rows in the vector so their
    // internal index buffers can be reused on later predictions.
    auto& path = _temporary.path;
    if (path.empty()) path.emplace_back();
    path[0].clear();
    path[0] = _sample;
    std::size_t path_size = 1;

    if (model_.nodes_.empty() || config_.args.size() > 8) return output;

    // A valid deserialized tree cannot revisit a node. This guard also keeps
    // prediction finite for a malformed model created in memory.
    auto& visited = _temporary.visited;
    std::size_t node_index = 0;

    while (true) {
        if (node_index >= model_.nodes_.size() || visited[node_index]) {
            return {};
        }
        visited[node_index] = true;

        const decision_tree::node& node = model_.nodes_[node_index];

        // A terminal node has no child references and points into leafs_.
        if (node.left == 0 && node.right == 0) {
            if (node.index < 0 ||
                static_cast<std::size_t>(node.index) >= model_.leafs_.size()) {
                return {};
            }

            const decision_tree::leaf& leaf =
                model_.leafs_[static_cast<std::size_t>(node.index)];

            // Each leaf coordinate stores a path row in `.x` and the target /
            // configuration row in `.y`. Store compatible hashes in the data
            // row selected by `.y`; every row must contribute at least one hash
            // or a complete label cannot be constructed.
            auto& data = _temporary.data;
            if (leaf.args.size() != config_.args.size()) return {};

            for (std::size_t i = 0; i < config_.args.size(); ++i) {
                const int path_index = leaf.args[i].x;
                const int target_index = leaf.args[i].y;
                if (path_index < 0 ||
                    static_cast<std::size_t>(path_index) >= path_size ||
                    target_index < 0 ||
                    static_cast<std::size_t>(target_index) >= config_.args.size()) {
                    return {};
                }

                auto& target_data = data[static_cast<std::size_t>(target_index)];
                for (int y_index : config_.args[static_cast<std::size_t>(target_index)]) {
                    if (y_index < 0 ||
                        static_cast<std::size_t>(y_index) >= config_.y.size()) {
                        return {};
                    }

                    set_intersect(path[static_cast<std::size_t>(path_index)],
                                  config_.y[static_cast<std::size_t>(y_index)].indexes,
                                  _temporary.intersection);
                    if (!_temporary.intersection.empty()) {
                        target_data.push_back(
                            config_.y[static_cast<std::size_t>(y_index)].hash);
                    }
                }
            }

            // An empty target row would produce incomplete label arguments, so
            // abort the entire prediction as requested.
            for (const auto& target_data : data) {
                if (target_data.empty()) return {};
            }

            // Choose exactly one hash from every data row. The recursive
            // expansion produces the Cartesian product of all compatible
            // choices. Every complete label receives exactly leaf.weight.
            label predicted;
            auto append_combinations = [&](auto&& self, std::size_t row) -> void {
                if (row == data.size()) {
                    output.add_prediction(predicted, leaf.weight);
                    return;
                }

                for (std::uint64_t hash : data[row]) {
                    predicted.args[row] = hash;
                    self(self, row + 1);
                }
            };
            append_combinations(append_combinations, 0);
            return output;
        }

        // An internal node must have two positive, in-range child indexes.
        if (node.left <= 0 || node.right <= 0 ||
            static_cast<std::size_t>(node.left) >= model_.nodes_.size() ||
            static_cast<std::size_t>(node.right) >= model_.nodes_.size() ||
            node.index < 0 ||
            static_cast<std::size_t>(node.index) >= path_size ||
            node.group < 0 ||
            static_cast<std::size_t>(node.group) >= config_.x.size()) {
            return {};
        }

        // Keep the intersection in an independent value because it is appended
        // to `path` on the left branch. This avoids references to a scratch
        // buffer being invalidated by vector growth or later intersections.
        set_intersect(path[static_cast<std::size_t>(node.index)],
                      config_.x[static_cast<std::size_t>(node.group)].indexes,
                      _temporary.next_path);

        if (_temporary.next_path.empty()) {
            node_index = static_cast<std::size_t>(node.right);
        } else {
            // Reuse an existing path row when possible. Swapping transfers the
            // newly computed indexes into the active row and returns that
            // row's old capacity to next_path for the next intersection.
            if (path_size == path.size()) path.emplace_back();
            path[path_size].clear();
            path[path_size].swap(_temporary.next_path);
            ++path_size;
            node_index = static_cast<std::size_t>(node.left);
        }
    }
}

} // namespace PhantomX
