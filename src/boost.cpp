// ============================================================================
//  boost.cpp
// ============================================================================
//
//  Implements binary serialization for PhantomX::boost, the top-level model
//  containing an ordered ensemble of decision trees.
//
//  This file does not implement gradient-boosting training or prediction.
//  Those responsibilities belong to boost::learner and boost::thinker. The
//  model container implemented here persists and restores only the trained
//  goal identifier and tree ensemble.
//
//  --------------------------------------------------------------------------
//  Binary format, version 2
//  --------------------------------------------------------------------------
//
//  The complete file is laid out as:
//
//      magic bytes              "PHANTOMX-BOOST"
//      uint32                  format version
//      uint64                  payload size
//      byte[payload size]      serialized boost payload
//      byte[32]                SHA-256 digest of the payload
//
//  The payload contains the goal identifier, tree count, and the serialized
//  trees in ensemble order. Tree order is significant because boosting applies
//  trees sequentially; swapping two otherwise identical trees changes the
//  resulting model.
//
//  Each embedded decision tree is itself self-delimiting and authenticated:
//  it has its own magic, version, payload size, and SHA-256 digest. This lets
//  the outer reader consume exactly one tree at a time while the inner reader
//  verifies that tree's contents independently.
//
//  The outer SHA-256 covers the complete boost payload, including the goal ID,
//  tree count, and every serialized tree byte. It detects corruption and
//  unexpected changes, but SHA-256 alone does not authenticate the producer:
//  someone who can replace a model can also recalculate its public digest. A
//  MAC or digital signature is required when producer authenticity matters.
//
//  Deserialization is transactional. The incoming goal ID and tree vector are
//  held in temporary values until every tree has been read, the payload has no
//  trailing bytes, and the complete model has passed verification. A failed
//  read does not partially replace the existing model.
//
//  The format stores raw fixed-width integer and floating-point representations.
//  It is intended as a compact local model format, not a normalized
//  cross-platform wire protocol.
//
// ============================================================================

#include <boost.hpp>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <picosha2/picosha2.h>

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
// Version 2 adds a payload-size field and a SHA-256 digest after the
// authenticated payload.
constexpr std::uint32_t format_version = 2;
constexpr std::size_t digest_size = picosha2::k_digest_size;

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

// Calculates the SHA-256 digest of the exact bytes that will be stored as
// the model payload.
std::vector<unsigned char> digest(const std::string& payload)
{
    std::vector<unsigned char> result(digest_size);
    picosha2::hash256(payload.begin(), payload.end(), result.begin(), result.end());
    return result;
}

bool write_digest(std::ostream& os, const std::string& payload)
{
    const auto value = digest(payload);
    os.write(reinterpret_cast<const char*>(value.data()),
            static_cast<std::streamsize>(value.size()));
    return static_cast<bool>(os);
}

bool read_and_verify_digest(std::istream& is, const std::string& payload)
{
    std::vector<unsigned char> expected(digest_size);
    is.read(reinterpret_cast<char*>(expected.data()),
            static_cast<std::streamsize>(expected.size()));
    if (!is) return false;

    const auto actual = digest(payload);
    return std::equal(expected.begin(), expected.end(), actual.begin());
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
//   uint64 payload size
//   payload: uint64 goal ID, uint64 tree count, serialized trees
//   32-byte SHA-256 digest of the payload
//
// The order of trees matters because boosting combines them sequentially. The
// first tree is not interchangeable with the last tree, even if both trees
// have the same internal structure.
bool boost::write(std::ostream& _os) const
{
    // Write the fixed signature without its terminating null character.
    // This is a byte-level file marker, not a stored C string.
    if (!_os.write(magic, sizeof(magic) - 1)) return false;

    // Build the complete payload first. This makes the exact bytes covered
    // by SHA-256 explicit and lets the outer reader verify the ensemble
    // before changing the current model.
    std::ostringstream payload_stream(std::ios::out | std::ios::binary);

    // goal_id_ identifies the target/goal represented by this ensemble.
    if (!write_value(payload_stream, goal_id_)) return false;

    // The count allows read() to know exactly how many decision_tree
    // records follow. uint64_t avoids making the file layout depend on
    // the platform-specific width of size_t.
    const auto tree_count = static_cast<std::uint64_t>(ensemble_.size());
    if (!write_value(payload_stream, tree_count)) return false;

    // Each decision tree owns its own serialization format. Calling
    // decision_tree::write() here keeps the two model layers independent:
    // boost describes the ensemble, while decision_tree describes one
    // member of that ensemble.
    for (const decision_tree& tree : ensemble_) {
        if (!tree.write(payload_stream)) return false;
    }

    const std::string payload = payload_stream.str();
    const auto payload_size = static_cast<std::uint64_t>(payload.size());

    if (!write_value(_os, format_version) ||
        !write_value(_os, payload_size)) return false;
    _os.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!_os) return false;
    return write_digest(_os, payload);
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

    // Read the payload size before allocating. The goal ID and tree count
    // themselves are inside the authenticated payload.
    std::uint32_t version = 0;
    std::uint64_t payload_size = 0;

    // Reject unsupported versions before reading tree records. A different
    // version may use a different record layout, so attempting to continue
    // would risk interpreting arbitrary bytes as model data.
    if (!read_value(_is, version) || version != format_version ||
        !read_value(_is, payload_size) || payload_size > std::string{}.max_size()) {
        return false;
    }

    // Read exactly this model's payload and verify it before interpreting
    // any goal or tree count supplied by the file.
    std::string payload(static_cast<std::size_t>(payload_size), '\0');
    _is.read(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!_is || !read_and_verify_digest(_is, payload)) return false;

    std::istringstream payload_stream(payload, std::ios::in | std::ios::binary);

    std::uint64_t loaded_goal_id = 0;
    std::uint64_t tree_count = 0;
    if (!read_value(payload_stream, loaded_goal_id) ||
        !read_value(payload_stream, tree_count) ||
        tree_count > ensemble_.max_size()) {
        return false;
    }

    // Keep the incoming ensemble separate from the current one until the
    // entire stream has been validated and decoded.
    std::vector<decision_tree> loaded_trees;
    loaded_trees.reserve(static_cast<std::size_t>(tree_count));

    // Trees are read in exactly the order in which write() stored them.
    // decision_tree::read() consumes precisely one serialized tree, so the
    // next iteration starts at the next tree boundary in the stream.
    for (std::uint64_t i = 0; i < tree_count; ++i) {
        decision_tree tree;
        if (!tree.read(payload_stream)) return false;
        loaded_trees.push_back(std::move(tree));
    }

    // No bytes may remain after the declared tree sequence.
    if (payload_stream.peek() != std::char_traits<char>::eof()) return false;

    // Commit the complete model only after every tree has been loaded. If
    // any earlier operation returned false, the old goal ID and tree vector
    // remain untouched.
    goal_id_ = loaded_goal_id;
    ensemble_ = std::move(loaded_trees);
    return true;
}

} // namespace PhantomX
