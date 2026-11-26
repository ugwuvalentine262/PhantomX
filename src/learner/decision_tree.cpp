// ============================================================================
//  decision_tree.cpp
// ============================================================================
//
//  Implementation of `decision_tree::learner`, declared in the learner header.
//
//  High-level picture
//  ------------------
//  The learner builds a decision tree over a dataset in which every sample is
//  described by a SET of attribute indexes (`sample.x`, a sorted vector<int>),
//  and by a number of targets (`sample.y[h]`, one per target h).
//
//  Each sample keeps a "decision path" (state::decision_path): the list of
//  groups of attributes it has been narrowed down to so far. The first group
//  of a path is the sample's full attribute set; every accepted split appends
//  one more group, formed by intersecting an existing group of the path with a
//  feature group from the configuration (config_.x[group]).
//
//  At each node, recursive_greedy_splitting():
//    1. applies the split that led to this node to the state (apply()),
//    2. randomly selects up to max_features_ feature groups when that limit is
//       enabled, tries each selected group against every path position with
//       divide(), and keeps the best split found,
//    3. if the best split improves on the score of the current branch, it
//       recurses into the left and right branches; otherwise this node
//       becomes a leaf,
//    4. rolls the state back (undo()) before returning.
//
//  divide() scores a candidate split by building, for each of the two
//  branches, a matrix of sample sets (rows = leaf groups of the path,
//  columns = targets) and asking Objective::solve() for the best selection of
//  cells with the largest common intersection. See divide() for details.
//
//  Conventions used throughout
//  ---------------------------
//    * `indexes` is a std::vector<int> that is assumed to be sorted,
//      non-negative and free of repeated values.
//    * `a & b` on two `indexes` returns the intersection of both vectors.
//    * A score of -1 means "not computed / rejected". A split with score -1
//      can never win against another split.
//    * "true branch" == left branch == samples that share at least one
//      attribute with the chosen feature group. "false branch" == right
//      branch == samples that share none.
//    * If max_features_ is positive, a random subset of at most
//      max_features_ feature groups is considered at each node. If it is
//      non-positive, every configured feature group is considered.
//
// ============================================================================

#include <algorithm>  // std::shuffle
#include <cmath>      // std::exp
#include <numeric>    // std::iota
#include <random>     // random feature selection

#include "decision_tree.hpp"  // the learner declaration (and decision_tree)
#include "objective.hpp"      // Objective::solve, Objective::Matrix

namespace PhantomX {

// ============================================================================
// recursive_greedy_splitting
// ============================================================================

// The recursive greedy splitting algorithm is the core of the decision tree
// learning algorithm. It divides the samples of a branch into smaller subsets
// using the feature group that best separates them, and then recurses into
// each subset.
//
// Parameters:
//   _state  - the decisions made so far on every sample of the dataset. It is
//             modified by apply() on entry and restored by undo() on exit.
//   _split  - the split that produced `_branch` (a default split for the root).
//   _branch - the branch (samples, score, args) this call is responsible for.
//
// Returns the index of the node created for this call. The index is the
// position of the node in the nodes_ vector, which is what lets a parent
// record where both of its children are located.
//
// Node indexes are handed out in pre-order (a node gets its index BEFORE its
// children are built), so a parent's index is always smaller than the indexes
// of its descendants and the root always has index 0.
int decision_tree::learner::recursive_greedy_splitting(state& _state,
                                                       const split& _split,
                                                       const branch& _branch) {
    // Claim the next free node index for this call.
    int node_idx = node_idx_++;

    // The best split found so far. A default split has score -1, so any
    // successfully scored split beats it.
    split best_split;

    // Narrow the state down to the branch we are currently working on.
    _state.apply(config_, _split, _branch);

    // Build the feature-group candidates for this node. Feature groups are
    // sampled without replacement when max_features_ is enabled, so one group
    // cannot be evaluated twice at the same node.
    //
    // The path length is read from the first sample of the branch. Paths of
    // samples in different branches can have different lengths (apply() only
    // extends the paths of the left branch), but all samples inside one
    // branch share the same path structure, so any of them can serve as the
    // reference. Requires `_branch.samples` to be non-empty.
    std::vector<std::size_t> feature_groups(config_.x.size());
    std::iota(feature_groups.begin(), feature_groups.end(), 0);

    if (max_features_ > 0 &&
        static_cast<std::size_t>(max_features_) < feature_groups.size()) {
        // A thread-local engine is seeded once per thread. This keeps the
        // selection random across nodes without reconstructing an engine for
        // every recursive call.
        static thread_local std::mt19937 random_engine(std::random_device{}());
        std::shuffle(feature_groups.begin(), feature_groups.end(), random_engine);
        feature_groups.resize(static_cast<std::size_t>(max_features_));
    }

    // Evaluate every selected feature group against every position in the
    // decision path, and retain the highest-scoring split.
    for (std::size_t g : feature_groups) {
        for (std::size_t i = 0; i < _state.paths[_branch.samples.front()].size(); i++) {
            // (The local variable intentionally shadows the type name `split`.)
            split split = divide(_branch.samples, _state, g, i);
            if (split.score > best_split.score) {
                best_split = std::move(split);
            }
        }
    }

    // Did we find a good split? A split is only accepted if the combined score
    // of its two branches is strictly greater than the score of the branch
    // being split. For the root, _branch.score is -1, so any scored split
    // is accepted.
    if (best_split.score > _branch.score) {
        // The evaluation order of the two recursive calls is unspecified in
        // C++. This does not matter: each call returns its own node index, so
        // the tree is correct whichever branch is built first.
        add_node(node_idx,
                 recursive_greedy_splitting(_state, best_split, best_split.left),
                 recursive_greedy_splitting(_state, best_split, best_split.right),
                 best_split);
    } else {
        // No good split was found: this branch becomes a leaf.
        add_leaf(node_idx, _branch);
    }

    // Restore the state so the caller can continue exploring other branches.
    _state.undo(_split, _branch);
    return node_idx;
}

// ============================================================================
// add_node / add_leaf
// ============================================================================

// Adds an internal node to the decision-tree model.
//
// Parameters:
//   _node_idx   - index of the node being written (position in nodes_).
//   _left_node  - index (in nodes_) of the left child.
//   _right_node - index (in nodes_) of the right child.
//   _split      - the split performed at this node; it supplies the feature
//                 group and the path position the split was made on.
void decision_tree::learner::add_node(int _node_idx,
                                      int _left_node,
                                      int _right_node,
                                      const split& _split) {
    // Children are built before their parent is written, so nodes_ may
    // already be larger than _node_idx. Grow it only if it is not.
    if (static_cast<std::size_t>(_node_idx) >= nodes_.size()) {
        nodes_.resize(_node_idx + 1);
    }

    auto& node = nodes_[_node_idx];

    node.left = _left_node;
    node.right = _right_node;
    node.group = _split.group;  // which feature group the node tests
    node.index = _split.index;  // which path position the test applies to
}

// Adds a leaf node to the decision-tree model.
//
// A leaf is stored twice: as an entry in nodes_ (so parents can point to it)
// and as an entry in leafs_ (which holds its `args` and its weight). The node
// refers to its leaf record through `node.index`.
void decision_tree::learner::add_leaf(int _node_idx, const branch& _branch) {
    if (static_cast<std::size_t>(_node_idx) >= nodes_.size()) {
        nodes_.resize(_node_idx + 1);
    }

    auto& node = nodes_[_node_idx];

    // A leaf has no children. Index 0 is the root, which can never be a child,
    // so 0 serves as the "no child" marker.
    node.left = 0;
    node.right = 0;
    node.group = 0;

    // For a leaf, `index` is the position of its record in leafs_. The record
    // is appended below, so its position is the current size.
    node.index = leafs_.size();

    // The weight of a leaf depends on the purity of the branch:
    //
    //     purity = score / number of samples          (purity in (0, 1])
    //     weight = e^(growth_ * purity) - 1
    //
    // The score is the size of the common intersection found by the objective
    // function, so purity is the fraction of the branch's samples that belong
    // to that intersection.
    //
    // _branch.score must be positive at this point, and the number of samples
    // must be non-zero (otherwise this is a division by zero).
    double purity = _branch.score / static_cast<double>(_branch.samples.size());

    leafs_.emplace_back(decision_tree::leaf{
        .args = _branch.args,
        .weight = std::exp(growth_ * purity) - 1.0,
    });
}

// ============================================================================
// state
// ============================================================================

// Updates the state to reflect that `_split` was applied for `_branch`.
//
// Only the left ("true") branch changes the state: its samples are narrowed
// down to the group obtained by intersecting the group at path position
// `_split.index` with the feature group `_config.x[_split.group]`. Samples in
// the right ("false") branch keep their current paths, so nothing is done.
//
// Precondition: every sample's path must have more than `_split.index`
// entries.
void decision_tree::learner::state::apply(const config& _config,
                                          const decision_tree::learner::split& _split,
                                          const branch& _branch) {
    // Right branch (and the root, which has true_branch == false): no change.
    if (!_branch.true_branch) {
        return;
    }

    // Extend the path of every sample in the branch by one group: the
    // intersection of the feature group (a) with the group being split (b).
    for (auto& i : _branch.samples) {
        auto& p = paths.at(i);
        auto& a = _config.x.at(_split.group);
        auto& b = p.at(_split.index);

        // The new group's constructor marks `b` as no longer being a leaf and
        // remembers `b`'s POSITION in the path (`_split.index`) as its parent,
        // but only if `b` was a leaf. A position, unlike a pointer, stays
        // valid when the vector `p` reallocates while it grows.
        //
        // The group is built first and appended afterwards, so `b` is never
        // read from the vector while that vector is being grown.
        group g(a, b, _split.index);
        p.push_back(std::move(g));
    }
}

// Retracts the split that was applied to the state: the state returns to what
// it was before apply() was called.
//
// For every sample of a left ("true") branch, the group added by apply() is
// removed from its path, and the parent group it was formed from (if it
// recorded one in `parent`) becomes a leaf again. As in apply(), the right
// branch never changed the state, so nothing is undone for it.
void decision_tree::learner::state::undo(const decision_tree::learner::split& _split,
                                         const branch& _branch) {
    if (!_branch.true_branch) {
        return;
    }

    for (std::size_t i = 0; i < _branch.samples.size(); i++) {
        auto& p = paths[_branch.samples[i]];
        const auto& q = p.back();  // the group appended by apply()

        // `q.parent` is a valid position (>= 0) only if the parent was a leaf
        // at the time q was created; in that case the parent regains its leaf
        // status. It is restored before q is removed, because q is read here.
        if (q.parent >= 0) {
            p[q.parent].leaf = true;
        }
        p.pop_back();
    }
}

// Constructor: initializes the state from a dataset.
//
// Every sample starts with a path that holds a single group: the sample's own
// attribute set (`sample.x`).
decision_tree::learner::state::state(const dataset& _dataset) {
    paths.reserve(_dataset.size());
    for (auto& sample : _dataset.samples) {
        paths.emplace_back(decision_path{group(sample.x)});
    }
}

// ============================================================================
// state::group
// ============================================================================

// Constructor.
// The new group is formed from the intersection of the specified groups: the
// feature group `_a` (from the configuration) and the existing group `_b`.
//
// The new group is initially a leaf, and `_b` stops being one.
//
// `_b_index` is the position of `_b` in the decision path. The new group
// remembers it in `parent` ONLY when `_b` was a leaf (otherwise `parent` is
// -1), so undo() knows whether it must make `_b` a leaf again. The check on
// `_b.leaf` happens in the initializer list, before the body marks `_b` as a
// non-leaf.
decision_tree::learner::state::group::group(const config::group& _a,
                                            decision_tree::learner::state::group& _b,
                                            int _b_index)
    : parent(_b.leaf ? _b_index : -1),
      indices(_a.indexes & _b.indices),
      leaf(true) {
    _b.leaf = false;
}

// Constructor.
// Initializes a root group from a set of indexes. It has no parent
// (`parent` is -1) and is initially a leaf.
decision_tree::learner::state::group::group(const indexes& _indexes)
    : parent(-1),
      indices(_indexes),
      leaf(true) {}

// ============================================================================
// divide
// ============================================================================

// Splits the given samples using the specified feature group (`_group`) at the
// specified path position (`_index`), and returns the resulting split.
//
// How a sample is routed:
//   intersect = (attributes of the sample's group at `_index`)
//               & (attributes of feature group `_group`)
//   - intersect is empty     -> the sample goes to the RIGHT ("false") branch
//   - intersect is non-empty -> the sample goes to the LEFT ("true") branch
//
// How a branch is scored:
//   For each branch, a matrix is built where
//     row    = one leaf group of the path AS IT WOULD BE after the split,
//     column = one target h (there are `k = config_.args.size()` targets),
//     cell   = the samples that were routed to this branch and whose target h
//              shares at least one index with the row's group.
//   Right branch: the rows are the current leaf groups of the path, unchanged
//   (no new group is created for it).
//   Left branch: the rows are the current leaf groups EXCEPT the group at
//   `_index` (if it is a leaf), followed by one last row for the new group
//   formed by `intersect`, which replaces it. If the group at `_index` is not
//   a leaf, nothing is excluded and the new group is simply one extra row.
//
//   Objective::solve() only sees the matrix, so the position of each row's
//   group in the decision path is lost. To recover it, one vector of path
//   positions is kept per branch (`right_leafs` and `left_leafs`): entry r is
//   the position, in the decision path, of the group that row r stands for.
//   After solving, the row (`.x`) of every selected cell is replaced by that
//   position, so the branch's `args` refer to groups of the decision path
//   instead of matrix rows. The column (`.y`, the target) is left as is.
//   The new left group is not in the path yet, so it is given the position it
//   will have once apply() appends it: the current length of the path.
//
//   Objective::solve() selects one cell per requested row (no two in the same
//   column) so that the intersection of the selected cells is as large as
//   possible. The learner requests min(k, number of rows) cells, where k is
//   the number of target columns. Therefore every row is represented when
//   there are enough target columns; when there are more rows than columns,
//   the objective performs a partial matching and some rows are intentionally
//   left unassigned. The size of the common intersection is the branch score,
//   and the selected cells become the branch's `args`. The score of the split
//   is the sum of the scores of its two branches.
//
// A rejected split (too few samples, an empty branch, a branch below the
// minimum leaf size) is returned with the default score of -1, which can
// never beat another split.
decision_tree::learner::split decision_tree::learner::divide(const indexes& _samples,
                                                             const state& _state,
                                                             int _group,
                                                             int _index) const {
    split split;

    // The number of samples at the node must reach the minimum required for
    // splitting. A non-positive min_samples_split_ disables this check.
    if (min_samples_split_ > 0 &&
        static_cast<std::size_t>(min_samples_split_) > _samples.size()) {
        return split;
    }

    // The path of the first sample is used as a representative for the whole
    // branch: all samples in `_samples` are assumed to share the same leaf
    // structure. Requires `_samples` to be non-empty.
    auto& first = _state.paths.at(_samples.front());

    // Number of targets (columns of the matrices).
    int k = config_.args.size();

    // Positions, in the decision path, of the groups that make up the rows of
    // each matrix. Entry r belongs to row r.
    //   right_leafs: every leaf group of the path.
    //   left_leafs:  every leaf group of the path except the one at `_index`,
    //                followed by the new group (added after the loop).
    std::vector<int> right_leafs, left_leafs;
    for (int i = 0; i < static_cast<int>(first.size()); i++) {
        if (!first[i].leaf) {
            continue;
        }
        right_leafs.push_back(i);

        // The group at `_index` is the one being split. On the left branch it
        // is replaced by the new group, so it gets no row of its own. (If it
        // is not a leaf it was skipped above, so nothing is excluded.)
        if (i != _index) {
            left_leafs.push_back(i);
        }
    }

    // The new group formed by `intersect` is a leaf of the left branch and
    // always takes the last row. It is not in the path yet; once apply() adds
    // it, it will sit at the end, i.e. at the current path length.
    left_leafs.push_back(static_cast<int>(first.size()));

    // The matrices handed to Objective::solve(): one row per entry of the
    // matching vector above, one column per target.
    Objective::Matrix left(left_leafs.size(), std::vector<indexes>(k));
    Objective::Matrix right(right_leafs.size(), std::vector<indexes>(k));

    // Adds `_sample` to every cell of row `_row` of `_matrix` whose column
    // (target h) is compatible with the group whose indexes are `_indices`:
    // the group of the sample's target h must share at least one index with it.
    auto add_to_row = [&](Objective::Matrix& _matrix,
                          std::size_t _row,
                          int _sample,
                          const indexes& _indices) {
        for (int h = 0; h < k; h++) {
            if (!(config_.y[dataset_[_sample].y[h]].indexes & _indices).empty()) {
                _matrix[_row][h].push_back(_sample);
            }
        }
    };

    for (auto sample : _samples) {
        auto& path = _state.paths[sample];

        // The attributes this sample has in common with the feature group.
        auto intersect = path[_index].indices & config_.x[_group].indexes;

        // Route the sample (once) to the right branch if it shares no
        // attribute with the feature group, otherwise to the left branch,
        // and add it to the rows of that branch's matrix.
        if (intersect.empty()) {
            split.right.samples.push_back(sample);

            // One row per leaf group of the path.
            for (std::size_t row = 0; row < right_leafs.size(); row++) {
                add_to_row(right, row, sample, path[right_leafs[row]].indices);
            }
        } else {
            split.left.samples.push_back(sample);

            // All rows but the last stand for existing leaf groups of the path.
            for (std::size_t row = 0; row + 1 < left_leafs.size(); row++) {
                add_to_row(left, row, sample, path[left_leafs[row]].indices);
            }

            // The last row stands for the new group, whose indexes are
            // `intersect`.
            add_to_row(left, left_leafs.size() - 1, sample, intersect);
        }
    }

    // A split that leaves one side without samples does not split anything.
    if (split.right.samples.empty() || split.left.samples.empty()) {
        return split;
    }

    // Each side must hold at least min_samples_leaf_ samples. A non-positive
    // min_samples_leaf_ disables this check.
    if (min_samples_leaf_ > 0) {
        if (static_cast<std::size_t>(min_samples_leaf_) > split.right.samples.size() ||
            static_cast<std::size_t>(min_samples_leaf_) > split.left.samples.size()) {
            return split;
        }
    }

    // Score both branches. solve() picks exactly min(rows, k) cells with
    // distinct rows and columns. This intentionally permits a partial
    // matching when the path has more leaf groups than configured target
    // columns. If no valid selection exists it returns an empty result (score
    // 0, nothing selected).
    auto r_res = Objective::solve(right, std::min(k, static_cast<int>(right_leafs.size())));
    auto l_res = Objective::solve(left, std::min(k, static_cast<int>(left_leafs.size())));

    // Replaces the row (`.x`) of every selected cell by the position, in the
    // decision path, of the group that row stood for. The column (`.y`) is
    // untouched.
    auto to_path_positions = [](coordinates& _selected, const std::vector<int>& _leafs) {
        for (auto& cell : _selected) {
            cell.x = _leafs[cell.x];
        }
    };
    to_path_positions(r_res.selectedIndices, right_leafs);
    to_path_positions(l_res.selectedIndices, left_leafs);

    // Right (false) branch.
    split.right.score = r_res.intersectionSize;
    split.right.args = r_res.selectedIndices;
    split.right.true_branch = false;

    // Left (true) branch.
    split.left.score = l_res.intersectionSize;
    split.left.args = l_res.selectedIndices;
    split.left.true_branch = true;

    // Quality of the whole split, and what it was made on.
    split.score = split.left.score + split.right.score;
    split.index = _index;
    split.group = _group;

    return split;
}

// ============================================================================
// learn
// ============================================================================

// Calls the main learning algorithm for the decision-tree model and returns
// the learned tree.
//
// `_samples` holds the indexes of the dataset samples to train on. The
// returned decision_tree holds the learned nodes and leaves (the root is
// node 0).
//
// The tree is built in the learner's own nodes_ and leafs_, which are then
// moved into the returned object. This is intentional: moving transfers
// ownership without copying the model data. The learner may be reused after
// the move; its working vectors are valid moved-from containers and are
// populated by the next training call. The hyper-parameters are kept.
decision_tree decision_tree::learner::learn(indexes _samples) {
    // The root has no parent split, so a default (empty) split is used.
    split null_split;

    // Every sample starts with a path made only of its own attributes.
    state state(dataset_);

    // Reset the node numbering. The working vectors were transferred to the
    // previous returned tree by std::move, so no explicit clear() is needed.
    node_idx_ = 0;

    // The root branch holds all the training samples. true_branch is false so
    // apply()/undo() leave the state untouched for the root.
    branch branch{.samples = std::move(_samples), .true_branch = false};

    // Build the tree. The returned node index of the root is always 0, so it
    // is discarded.
    (void)recursive_greedy_splitting(state, null_split, branch);

    // Hand the finished nodes and leaves over to the tree to return. The
    // learner is a nested class of decision_tree, so it may access the tree's
    // private members. Moving avoids copying; the learner's nodes_ and leafs_
    // are left in valid moved-from states and are reused by the next call
    // without copying the completed model.
    decision_tree tree;
    tree.nodes_ = std::move(nodes_);
    tree.leafs_ = std::move(leafs_);
    return tree;
}

// ============================================================================
// Constructor
// ============================================================================

// The learning algorithm is initialized with a dataset and a
// configuration/grouping of the attributes and features. The hyper-parameters
// keep their defaults (constraints disabled, growth_ = 2.5) until set through
// the set_* functions below.
decision_tree::learner::learner(const dataset& _dataset, const config& _config)
    : dataset_(_dataset),
      config_(_config) {}

// ============================================================================
// Hyper-parameter setters
// ============================================================================
//
// Each setter returns true if the value was accepted (and stored) and false if
// it was rejected. On rejection the member keeps its previous value.

// Sets the number of feature groups considered while looking for a split
// point. Rejected only when `i` is 0. A negative value is accepted and means
// "no limit". When positive, a random subset of at most this many feature
// groups is sampled independently at each tree node.
bool decision_tree::learner::set_max_features(int i) {
    if (i == 0) {
        return false;
    }
    max_features_ = i;
    return true;
}

// Sets the minimum number of samples expected at a leaf.
// Rejected only when `i` is 0. A negative value is accepted and disables the
// constraint (see divide()).
bool decision_tree::learner::set_min_samples_leaf(int i) {
    if (i == 0) {
        return false;
    }
    min_samples_leaf_ = i;
    return true;
}

// Sets the minimum number of samples a node needs before a split is
// considered.
// Rejected only when `i` is 0. A negative value is accepted and disables the
// constraint (see divide()).
bool decision_tree::learner::set_min_samples_split(int i) {
    if (i == 0) {
        return false;
    }
    min_samples_split_ = i;
    return true;
}

// Sets the exponential growth constant `k` used in f(x) = e^(k * x) - 1 to
// weight leaf nodes (see add_leaf()).
// Accepted only when `v` is strictly greater than 0. Written as `!(v > 0)` so
// that NaN is rejected as well.
bool decision_tree::learner::set_growth_rate(float v) {
    if (!(v > 0.0f)) {
        return false;
    }
    growth_ = v;
    return true;
}

}  // namespace PhantomX
