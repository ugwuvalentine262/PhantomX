// ============================================================================
//  decision_tree.cpp
// ============================================================================
//
//  Implements binary serialization for PhantomX::decision_tree.
//
//  This file intentionally does not train or predict. Those responsibilities
//  belong to decision_tree::learner and decision_tree::thinker. The model
//  container implemented here only converts its internal node and leaf vectors
//  to bytes and reconstructs them from bytes later.
//
//  --------------------------------------------------------------------------
//  Binary format, version 3
//  --------------------------------------------------------------------------
//
//  The complete file is laid out as:
//
//      magic bytes              "PHANTOMX-DT"
//      uint32                  format version
//      uint64                  payload size
//      byte[payload size]      serialized tree payload
//      byte[32]                SHA-256 digest of the payload
//
//  The payload contains the node count, leaf count, all node records, and all
//  leaf records. Each node stores left/right child indexes plus its split
//  index/group. Each leaf stores its coordinate arguments and floating weight.
//
//  The leading magic identifies the file type before any model data is parsed.
//  The version prevents a future layout from being silently interpreted as
//  this layout. The payload size makes one tree self-delimiting, which is
//  required when a tree is nested inside a serialized boost ensemble.
//
//  SHA-256 is calculated over the exact binary payload. Writing first builds
//  the payload, calculates its digest, then writes the payload followed by the
//  digest. Reading verifies the digest before parsing counts or allocating the
//  node/leaf vectors. This detects corruption and accidental modification, but
//  SHA-256 alone does not authenticate the producer: someone who can replace a
//  model can also recalculate its public digest. Authentication would require
//  a MAC or digital signature.
//
//  Deserialization is transactional. Nodes and leaves are loaded into temporary
//  vectors and committed only after the payload is complete, authenticated,
//  successfully parsed, and free of trailing bytes. A failed read therefore
//  leaves the existing tree unchanged.
//
//  The format stores raw fixed-width integer and floating-point representations.
//  It is intended as a compact local model format, not a normalized
//  cross-platform wire protocol.
//
// ============================================================================

#include <decision_tree.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <picosha2/picosha2.h>

namespace PhantomX {

namespace {

// -----------------------------------------------------------------------------
// Private serialization helpers
// -----------------------------------------------------------------------------
//
// Everything in this anonymous namespace is private to this implementation
// file. These helpers are deliberately not part of the public decision_tree
// API because callers only need decision_tree::write() and decision_tree::read().
// Keeping the low-level stream operations here also makes the public methods
// easier to read: they can describe the structure of the file rather than
// repeating byte-level stream code.

// A model file starts with a fixed signature. This prevents read() from
// silently interpreting an unrelated file as a decision tree.
constexpr char magic[] = "PHANTOMX-DT";

// The version allows the on-disk format to evolve later. If the layout changes,
// a new version can be introduced and old files can be rejected or migrated
// explicitly instead of being read incorrectly. Version 2 added a
// payload-size field and a SHA-256 digest after the payload. Version 3 stores
// leaf weights as doubles instead of floats.
constexpr std::uint32_t format_version = 3;
constexpr std::size_t digest_size = picosha2::k_digest_size;

// Writes one trivially-copyable value as raw bytes.
//
// The decision_tree node fields use fixed-width integer types, and the leaf
// weight is a double. This preserves the learner's computed precision in the
// local model format.
// not presented as a portable wire protocol because byte order and floating
// point representation are not normalized here.
template <typename T>
bool write_value(std::ostream& os, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
    return static_cast<bool>(os);
}

// Reads one trivially-copyable value from the stream and reports whether the
// complete value was read successfully. A short or failed read leaves the
// caller with false, allowing the public read() method to reject the file.
template <typename T>
bool read_value(std::istream& is, T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    return static_cast<bool>(is);
}

// Calculates the SHA-256 digest of a serialized payload. The digest is kept
// as raw 32-byte data rather than hexadecimal text so the file remains small
// and the comparison does not depend on text formatting.
std::vector<unsigned char> digest(const std::string& payload)
{
    std::vector<unsigned char> result(digest_size);
    picosha2::hash256(payload.begin(), payload.end(), result.begin(), result.end());
    return result;
}

// Validates the structural invariants of a decoded tree before it is exposed
// through the public model object.
//
// Node zero is the root. Internal nodes must point to two different, positive
// node indexes, and every non-root node must be referenced exactly once. A leaf
// node is identified by both child links being zero and must point to a unique,
// in-range entry in the separate leaf-record vector. Requiring every node and
// leaf record to be referenced prevents disconnected nodes, duplicate child
// references, dangling leaf indexes, and incomplete leaf tables.
bool is_valid_tree(const decision_tree::nodes& nodes,
                   const decision_tree::leafs& leafs)
{
    if (nodes.empty() || leafs.empty()) return false;

    std::vector<bool> child_seen(nodes.size(), false);
    std::vector<bool> leaf_seen(leafs.size(), false);

    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
        const auto& node = nodes[node_index];

        const bool left_is_zero = node.left == 0;
        const bool right_is_zero = node.right == 0;

        if (left_is_zero && right_is_zero) {
            // Terminal nodes do not have child references. Their index points
            // into the separately serialized leaf-record vector.
            if (node.index < 0 ||
                static_cast<std::size_t>(node.index) >= leafs.size() ||
                leaf_seen[static_cast<std::size_t>(node.index)]) {
                return false;
            }
            leaf_seen[static_cast<std::size_t>(node.index)] = true;
            continue;
        }

        // Internal nodes must have two non-zero child references. A zero on
        // only one side, equal child references, negative indexes, and indexes
        // beyond the node vector are all malformed.
        if (left_is_zero || right_is_zero || node.left == node.right ||
            node.left < 0 || node.right < 0 ||
            static_cast<std::size_t>(node.left) >= nodes.size() ||
            static_cast<std::size_t>(node.right) >= nodes.size()) {
            return false;
        }

        const auto left = static_cast<std::size_t>(node.left);
        const auto right = static_cast<std::size_t>(node.right);
        if (child_seen[left] || child_seen[right]) return false;
        child_seen[left] = true;
        child_seen[right] = true;
    }

    // The root has no parent, while every other node must be referenced once.
    if (child_seen[0]) return false;
    for (std::size_t node_index = 1; node_index < child_seen.size(); ++node_index) {
        if (!child_seen[node_index]) return false;
    }

    // Every serialized leaf record must correspond to exactly one leaf node.
    for (bool seen : leaf_seen) {
        if (!seen) return false;
    }

    return true;
}

// Writes a raw SHA-256 digest and verifies the stream state.
bool write_digest(std::ostream& os, const std::string& payload)
{
    const auto value = digest(payload);
    os.write(reinterpret_cast<const char*>(value.data()),
             static_cast<std::streamsize>(value.size()));
    return static_cast<bool>(os);
}

// Reads and verifies a raw SHA-256 digest for a payload.
bool read_and_verify_digest(std::istream& is, const std::string& payload)
{
    std::vector<unsigned char> expected(digest_size);
    is.read(reinterpret_cast<char*>(expected.data()),
            static_cast<std::streamsize>(expected.size()));
    if (!is) return false;

    const auto actual = digest(payload);
    return std::equal(expected.begin(), expected.end(), actual.begin());
}

// Coordinates are variable-length vectors, so they are stored as:
//
//   number of coordinates
//   x, y for coordinate 0
//   x, y for coordinate 1
//   ...
//
// The count is written as uint64_t so the file format does not depend on the
// platform's size_t width.
bool write_coordinates(std::ostream& os, const coordinates& values)
{
    const auto count = static_cast<std::uint64_t>(values.size());
    if (!write_value(os, count)) return false;

    for (const coordinate& value : values) {
        const auto x = static_cast<std::int32_t>(value.x);
        const auto y = static_cast<std::int32_t>(value.y);
        if (!write_value(os, x) || !write_value(os, y)) return false;
    }
    return true;
}

// Reads the variable-length coordinate representation described above.
//
// The values are first loaded into a temporary vector. The destination is
// updated only after the complete coordinate sequence has been read. This
// prevents a truncated stream from leaving a leaf partially modified.
bool read_coordinates(std::istream& is, coordinates& values)
{
    std::uint64_t count = 0;
    if (!read_value(is, count) || count > values.max_size()) return false;

    coordinates result;
    result.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        std::int32_t x = 0;
        std::int32_t y = 0;
        if (!read_value(is, x) || !read_value(is, y)) return false;
        result.push_back(coordinate{static_cast<int>(x), static_cast<int>(y)});
    }
    values = std::move(result);
    return true;
}

} // namespace

// -----------------------------------------------------------------------------
// Leaf serialization
// -----------------------------------------------------------------------------

// A leaf contains the coordinate arguments used to derive its target and a
// floating-point weight. The coordinate vector has its own length prefix;
// write_value() then stores the weight immediately after the coordinates.
bool decision_tree::leaf::write(std::ostream& _os) const
{
    return write_coordinates(_os, args) && write_value(_os, weight);
}

// Reads a leaf using temporary values so a failed read does not partially
// overwrite the leaf that the caller already owns.
bool decision_tree::leaf::read(std::istream& _is)
{
    coordinates loaded_args;
    double loaded_weight = 0.0;
    if (!read_coordinates(_is, loaded_args) || !read_value(_is, loaded_weight)) {
        return false;
    }
    args = std::move(loaded_args);
    weight = loaded_weight;
    return true;
}

// -----------------------------------------------------------------------------
// Decision-tree serialization
// -----------------------------------------------------------------------------

// Writes the nodes and leaf data of the decision-tree
// to the specified output-stream. Returns true iff the
// operation completed successfully.
//
// File layout, version 3:
//
//   magic bytes
//   uint32 version
//   uint64 payload size
//   payload: uint64 node count, uint64 leaf count, node array, leaf array
//   32-byte SHA-256 digest of the payload
//
// Nodes are written before leaves because nodes contain references to child
// nodes and leaf indexes. The vectors are serialized in their existing order,
// so all indexes remain meaningful after a round trip. A decoded tree is
// structurally validated before it replaces the current model.
bool decision_tree::write(std::ostream& _os) const
{
	// Write the signature without its terminating '\0'. The signature is a
    // sequence of file bytes, not a C string stored in the model file.
    if (!_os.write(magic, sizeof(magic) - 1)) return false;

    // The version is written immediately after the signature so read() can
    // reject unsupported layouts before allocating any model storage.
    // Build the complete payload in memory first. This gives us the exact byte
    // sequence that will be hashed and also lets the reader know exactly where
    // this tree ends when it is nested inside a boost ensemble.
    std::ostringstream payload_stream(std::ios::binary);

    // Use fixed-width counts rather than size_t, whose width can differ across
    // platforms. The casts are safe because vector::size() is size_t and the
    // target type is at least as wide on supported platforms.
    const auto node_count = static_cast<std::uint64_t>(nodes_.size());
    const auto leaf_count = static_cast<std::uint64_t>(leafs_.size());
    if (!write_value(payload_stream, node_count) ||
        !write_value(payload_stream, leaf_count)) return false;

	// A node is a fixed-size record. Its fields are written individually rather
	// than writing the struct as one block, which makes the intended file
	// layout explicit and avoids depending on compiler-inserted padding.
    for (const node& value : nodes_) {
        if (!write_value(payload_stream, value.left) ||
            !write_value(payload_stream, value.right) ||
            !write_value(payload_stream, value.index) ||
            !write_value(payload_stream, value.group)) {
            return false;
        }
    }

	// Leaves contain a variable-size coordinate vector, so each leaf delegates
	// its serialization to leaf::write().
    for (const leaf& value : leafs_) {
        if (!value.write(payload_stream)) return false;
    }

    const std::string payload = payload_stream.str();
    const auto payload_size = static_cast<std::uint64_t>(payload.size());

    if (!write_value(_os, format_version) ||
        !write_value(_os, payload_size)) return false;
    _os.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!_os) return false;
    return write_digest(_os, payload);
}

// Reads the nodes and leaf data of the decision-tree
// from the specified input-stream. Returns true iff the
// operation completed successfully.
//
// The method is transactional: it reads into local vectors and only moves them
// into nodes_ and leafs_ after every record succeeds. If the stream is invalid,
// truncated, has the wrong signature/version, or requests an impossible
// allocation, the existing tree remains unchanged.
bool decision_tree::read(std::istream& _is)
{
	// Read and validate the fixed-size signature first.
    char header[sizeof(magic) - 1] = {};
    if (!_is.read(header, sizeof(header))) return false;
    if (!std::equal(std::begin(header), std::end(header), magic)) return false;

    // Read the version and payload size before allocating. Unsupported versions
    // must be rejected because their record layout may differ.
    std::uint32_t version = 0;
    std::uint64_t payload_size = 0;
    if (!read_value(_is, version) || version != format_version ||
        !read_value(_is, payload_size) || payload_size > std::string{}.max_size()) {
        return false;
    }

	// Read exactly this tree's payload. The explicit size is what makes it safe
	// for boost::read() to place several serialized trees in one stream.
    std::string payload(static_cast<std::size_t>(payload_size), '\0');
    _is.read(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!_is || !read_and_verify_digest(_is, payload)) return false;

    std::istringstream payload_stream(payload, std::ios::binary);

	// Counts are inside the authenticated payload, so they are covered by the
	// SHA-256 verification before any vector allocation is attempted.
    std::uint64_t node_count = 0;
    std::uint64_t leaf_count = 0;
    if (!read_value(payload_stream, node_count) ||
        !read_value(payload_stream, leaf_count) ||
        node_count > nodes_.max_size() || leaf_count > leafs_.max_size()) {
        return false;
    }

	// These temporary vectors are the reason read() is transactional. Existing
	// model data is retained until the entire incoming model is valid.
    nodes loaded_nodes;
    leafs loaded_leafs;
    loaded_nodes.reserve(static_cast<std::size_t>(node_count));
    loaded_leafs.reserve(static_cast<std::size_t>(leaf_count));

	// Reconstruct every node in the same order used by write().
    for (std::uint64_t i = 0; i < node_count; ++i) {
        node value{};
        if (!read_value(payload_stream, value.left) ||
            !read_value(payload_stream, value.right) ||
            !read_value(payload_stream, value.index) ||
            !read_value(payload_stream, value.group)) {
            return false;
        }
        loaded_nodes.push_back(value);
    }

	// Reconstruct every variable-size leaf after all nodes have been read.
    for (std::uint64_t i = 0; i < leaf_count; ++i) {
        leaf value {};
    if (!value.read(payload_stream)) return false;
        loaded_leafs.push_back(std::move(value));
    }

    // The payload must contain exactly the records described by its counts;
    // trailing bytes indicate a malformed or incompatible serialization.
    if (payload_stream.peek() != std::char_traits<char>::eof()) return false;

    // Validate all node/leaf references before committing the decoded model.
    // This keeps malformed structures out of the public object and preserves
    // the transactional behavior of read().
    if (!is_valid_tree(loaded_nodes, loaded_leafs)) return false;

    // Commit only after the complete stream has been decoded successfully and
    // the tree structure has passed validation.
    nodes_ = std::move(loaded_nodes);
    leafs_ = std::move(loaded_leafs);
    return true;
}

} // namespace PhantomX
