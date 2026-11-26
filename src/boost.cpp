#include <boost.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace PhantomX {

namespace {

    // -------------------------------------------------------------------------
    // Private serialization helpers
    // -------------------------------------------------------------------------
    //
    // These declarations live in an anonymous namespace because they are
    // implementation details of this source file. The public API remains the
    // two functions declared by boost: write() and read().

    // The boost model has its own file signature so it cannot be confused with
    // a serialized decision tree. A separate signature is especially useful
    // because a boost model contains decision trees internally, but is not
    // itself encoded as a single decision_tree file.
    constexpr char magic[] = "PHANTOMX-BOOST";

    // The version number identifies the layout of the bytes following magic.
    // If fields are added or reordered in a future release, the version should
    // be incremented so read() can reject an incompatible file safely.
    constexpr std::uint32_t format_version = 1;

    // Writes a trivially-copyable value directly to the output stream.
    //
    // The model currently uses fixed-width integer fields for the serialized
    // metadata. The helper is also used for goal_id_, which is uint64_t. Raw
    // binary output keeps the format compact, but does not normalize byte order
    // or floating-point representation for cross-platform interchange.
    template <typename T>
    bool write_value(std::ostream& os, const T& value)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        os.write(reinterpret_cast<const char*>(&value), sizeof(value));
        return static_cast<bool>(os);
    }

    // Reads one complete trivially-copyable value from the input stream.
    // Returning the stream state allows callers to detect truncated files and
    // other I/O failures immediately after each read operation.
    template <typename T>
    bool read_value(std::istream& is, T& value)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        is.read(reinterpret_cast<char*>(&value), sizeof(value));
        return static_cast<bool>(is);
    }

} // namespace

    // -----------------------------------------------------------------------------
    // Boost model serialization
    // -----------------------------------------------------------------------------

    // Writes the boosting model in this order:
    //
    //   magic bytes
    //   uint32 format version
    //   uint64 goal ID
    //   uint64 number of trees
    //   serialized decision trees in ensemble order
    //
    // The order of trees matters because boosting combines them sequentially. The
    // first tree is not interchangeable with the last tree, even if both trees
    // have the same internal structure.
    bool boost::write(std::ostream& _os) const
    {
        // Write the fixed signature without its terminating null character.
        // This is a byte-level file marker, not a stored C string.
        if (!_os.write(magic, sizeof(magic) - 1)) return false;

        // Store the version before any variable-size data. A reader can reject
        // an unsupported format before allocating the tree vector.
        if (!write_value(_os, format_version)) return false;

        // goal_id_ identifies the target/goal represented by this ensemble.
        // It belongs to the boost model rather than to an individual tree.
        if (!write_value(_os, goal_id_)) return false;

        // The count allows read() to know exactly how many decision_tree
        // records must follow. uint64_t avoids making the file layout depend
        // on the platform-specific width of size_t.
        const auto tree_count = static_cast<std::uint64_t>(ensemble_.size());
        if (!write_value(_os, tree_count)) return false;

        // Each decision tree owns its own serialization format. Calling
        // decision_tree::write() here keeps the two model layers independent:
        // boost describes the ensemble, while decision_tree describes one
        // member of that ensemble.
        for (const decision_tree& tree : ensemble_) {
            if (!tree.write(_os)) return false;
        }

        return true;
    }

    // Reads the same format produced by write(). The ensemble is first loaded into
    // a temporary vector, so a failed or truncated stream leaves the current model
    // unchanged. This is important when a boost object is reused to load several
    // model files: a failed second load must not leave it containing half of the
    // second model.
    bool boost::read(std::istream& _is)
    {
        // Read and validate the signature before interpreting any subsequent
        // bytes as version, goal ID, or collection counts.
        char header[sizeof(magic) - 1] = {};
        if (!_is.read(header, sizeof(header))) return false;
        if (!std::equal(std::begin(header), std::end(header), magic)) return false;

        // These values are read into locals first. In particular, goal_id_ is
        // not changed until all trees have been read successfully.
        std::uint32_t version = 0;
        std::uint64_t loaded_goal_id = 0;
        std::uint64_t tree_count = 0;

        // Reject unsupported versions before reading tree records. A different
        // version may use a different record layout, so attempting to continue
        // would risk interpreting arbitrary bytes as model data.
        if (!read_value(_is, version) || version != format_version ||
            !read_value(_is, loaded_goal_id) || !read_value(_is, tree_count)) {
            return false;
        }

        // The count comes from the input stream and must be checked before
        // reserve() is called. This prevents a malformed file from requesting
        // more vector capacity than the implementation can represent.
        if (tree_count > ensemble_.max_size()) return false;

        // Keep the incoming ensemble separate from the current one until the
        // entire stream has been validated and decoded.
        std::vector<decision_tree> loaded_trees;
        loaded_trees.reserve(static_cast<std::size_t>(tree_count));

        // Trees are read in exactly the order in which write() stored them.
        // decision_tree::read() consumes precisely one serialized tree, so the
        // next iteration starts at the next tree boundary in the stream.
        for (std::uint64_t i = 0; i < tree_count; ++i) {
            decision_tree tree;
            if (!tree.read(_is)) return false;
            loaded_trees.push_back(std::move(tree));
        }

        // Commit the complete model only after every tree has been loaded. If
        // any earlier operation returned false, the old goal ID and tree vector
        // remain untouched.
        goal_id_ = loaded_goal_id;
        ensemble_ = std::move(loaded_trees);
        return true;
    }

} // namespace PhantomX
