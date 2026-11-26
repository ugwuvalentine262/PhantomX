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
// explicitly instead of being read incorrectly. Version 2 adds a payload-size
// field and a SHA-256 digest after the payload.
constexpr std::uint32_t format_version = 2;
constexpr std::size_t digest_size = picosha2::k_digest_size;

// Writes one trivially-copyable value as raw bytes.
//
// The decision_tree node fields use fixed-width integer types, and the leaf
// weight is a float. This is a compact local model format. It is intentionally
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
    float loaded_weight = 0.0f;
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
// File layout, version 2:
//
//   magic bytes
//   uint32 version
//   uint64 payload size
//   payload: uint64 node count, uint64 leaf count, node array, leaf array
//   32-byte SHA-256 digest of the payload
//
// Nodes are written before leaves because nodes contain references to child
// nodes and leaf indexes. The vectors are serialized in their existing order,
// so all indexes remain meaningful after a round trip.
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

    // Commit only after the complete stream has been decoded successfully.
    nodes_ = std::move(loaded_nodes);
    leafs_ = std::move(loaded_leafs);
    return true;
}

} // namespace PhantomX
