#include <algorithm>
#include <functional>
#include <string>
#include <iostream>
#include <vector>
#include <iterator>
#include <limits>
#include <string>
#include <map>
#include <cstdint>
#include <set>
#include <utility>

#include <nlohmann/json.hpp>
#include <configuration.hpp>

namespace PhantomX {

    using json = nlohmann::json;

    // ---------------------------------------------------------------------------
    // Integer validation
    // ---------------------------------------------------------------------------

    // Checks that `value` is a JSON integer in the range [0, INT_MAX].
    // Booleans, null, strings, floats (even "3.0"), negatives and values above
    // INT_MAX are rejected. `where` describes the element for error messages.
    // Returns the validated value as int.
    static int validate_int_value(const json& value, const std::string& where) {
        // nlohmann::json keeps booleans as a distinct type, but check explicitly
        // so the intent is obvious.
        if (value.is_boolean()) {
            throw ValidationError(where + " must be a non-negative integer, got a boolean.");
        }
        if (value.is_null()) {
            throw ValidationError(where + " must be a non-negative integer, got null.");
        }
        if (value.is_string()) {
            throw ValidationError(where + " must be a non-negative integer, got a string.");
        }
        if (value.is_number_float()) {
            throw ValidationError(where + " must be an integer, got a floating-point number.");
        }

        if (value.is_number_unsigned()) {
            const std::uint64_t v = value.get<std::uint64_t>();
            if (v > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                throw ValidationError(where + " exceeds the maximum value of int (" +
                                      std::to_string(std::numeric_limits<int>::max()) + ").");
            }
            return static_cast<int>(v);
        }

        if (value.is_number_integer()) {
            // Stored as signed; only non-negative values are acceptable.
            const std::int64_t v = value.get<std::int64_t>();
            if (v < 0) {
                throw ValidationError(where + " must be non-negative, got " +
                                      std::to_string(v) + ".");
            }
            if (v > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
                throw ValidationError(where + " exceeds the maximum value of int (" +
                                      std::to_string(std::numeric_limits<int>::max()) + ").");
            }
            return static_cast<int>(v);
        }

        throw ValidationError(where + " must be a non-negative integer, got " +
                              std::string(value.type_name()) + ".");
    }

    // ---------------------------------------------------------------------------
    // Root validation
    // ---------------------------------------------------------------------------

    // The root must be an object containing "x", "y" and "args".
    static void validate_root(const json& root) {
        if (!root.is_object()) {
            throw ValidationError(std::string("Root must be a JSON object, got ") +
                                  root.type_name() + ".");
        }
        for (const char* key : {"x", "y", "args"}) {
            if (!root.contains(key)) {
                throw ValidationError(std::string("Root object is missing required key \"") +
                                      key + "\".");
            }
        }
    }

    // ---------------------------------------------------------------------------
    // "x" / "y" validation (identical rules)
    // ---------------------------------------------------------------------------

    // `name` is "x" or "y" and is used for error messages.
    // Returns the samples sorted by key hash; each Sample's integers are sorted
    // ascending. Rejects repeated integers within an array and hash collisions
    // between keys of this object.
    static config::groups validate_integer_map(const json& obj, const std::string& name) {

        config::groups samples;

        if (!obj.is_object()) {
            throw ValidationError("\"" + name + "\" must be a JSON object, got " +
                                  obj.type_name() + ".");
        }
        if (obj.empty()) {
            throw ValidationError("\"" + name + "\" must contain at least one key-value pair.");
        }

        samples.reserve(obj.size());
        // Temporary map hash -> key, used only to detect and report collisions.
        std::map<std::size_t, std::string> hash_owner;

        for (const auto& item : obj.items()) {
            const std::string& key = item.key();
            const json& arr = item.value();
            const std::string path = "\"" + name + "\"[\"" + key + "\"]";

            if (!arr.is_array()) {
                throw ValidationError("Value of " + path + " must be an array, got " +
                                      arr.type_name() + ".");
            }
            if (arr.empty()) {
                throw ValidationError("Array " + path + " must contain at least one element.");
            }
            std::vector<int> values;
            values.reserve(arr.size());
            for (std::size_t i = 0; i < arr.size(); ++i) {
                values.push_back(
                    validate_int_value(arr[i], "Element " + path + "[" + std::to_string(i) + "]"));
            }

            // Sort ascending, then any repeated value is adjacent to its twin.
            std::sort(values.begin(), values.end());
            const auto dup = std::adjacent_find(values.begin(), values.end());
            if (dup != values.end()) {
                throw ValidationError("Array " + path + " contains the value " +
                                      std::to_string(*dup) +
                                      " more than once; integers must not repeat within an array.");
            }

            const std::size_t hash = std::hash<std::string>{}(key);
            const auto inserted = hash_owner.emplace(hash, key);
            if (!inserted.second) {
                throw ValidationError("Hash collision in \"" + name + "\": keys \"" +
                                      inserted.first->second + "\" and \"" + key +
                                      "\" have the same std::hash value.");
            }

            samples.push_back(config::group{std::move(values), hash});
        }

        std::sort(samples.begin(), samples.end(),
                  [](const config::group& a, const config::group& b) { return a.hash < b.hash; });
        return samples;
    }

    // ---------------------------------------------------------------------------
    // Key separation
    // ---------------------------------------------------------------------------

    // No key may appear in both "x" and "y". Assumes both are already objects.
    static void validate_key_separation(const json& x, const json& y) {
        for (const auto& item : x.items()) {
            if (y.contains(item.key())) {
                throw ValidationError("Key \"" + item.key() +
                                      "\" appears in both \"x\" and \"y\"; keys must be mutually exclusive.");
            }
        }
    }

    // ---------------------------------------------------------------------------
    // "args" validation
    // ---------------------------------------------------------------------------

    // Returns the index of the sample for `key` in `sorted` (sorted by hash).
    // The key is known to exist in "y" and hashes are unique, so it must be found.
    static int find_sample_index(const config::groups& sorted, const std::string& key) {
        const std::size_t hash = std::hash<std::string>{}(key);
        const auto it = std::lower_bound(
            sorted.begin(), sorted.end(), hash,
            [](const config::group& s, std::size_t h) { return s.hash < h; });
        return static_cast<int>(it - sorted.begin());
    }

    // Every string must be a key of "y"; no duplicates within one inner array.
    // Returns, for each inner array, the indices of the corresponding samples in
    // the already sorted `y_samples`.
    static config::args_space validate_args(const json& args, const json& x, const json& y,
                        const config::groups& y_samples) {
        config::args_space result;

        if (!args.is_array()) {
            throw ValidationError(std::string("\"args\" must be a JSON array, got ") +
                                  args.type_name() + ".");
        }
        if (args.empty()) {
            throw ValidationError("\"args\" must contain at least one inner array.");
        }

        for (std::size_t i = 0; i < args.size(); ++i) {
            const json& inner = args[i];
            const std::string inner_path = "\"args\"[" + std::to_string(i) + "]";

            if (!inner.is_array()) {
                throw ValidationError("Element " + inner_path + " must be an array, got " +
                                      inner.type_name() + ".");
            }
            if (inner.empty()) {
                throw ValidationError("Array " + inner_path + " must contain at least one element.");
            }

            std::set<std::string> seen; // duplicate detection is per inner array
            std::vector<int> indices;
            indices.reserve(inner.size());

            for (std::size_t j = 0; j < inner.size(); ++j) {
                const json& elem = inner[j];
                const std::string elem_path = inner_path + "[" + std::to_string(j) + "]";

                if (!elem.is_string()) {
                    throw ValidationError("Element " + elem_path + " must be a string, got " +
                                          elem.type_name() + ".");
                }

                const std::string& s = elem.get_ref<const std::string&>();

                if (!y.contains(s)) {
                    if (x.contains(s)) {
                        throw ValidationError("Element " + elem_path + " (\"" + s +
                                              "\") is a key of \"x\", but only keys of \"y\" are allowed.");
                    }
                    throw ValidationError("Element " + elem_path + " (\"" + s +
                                          "\") does not match any key in \"y\".");
                }

                if (!seen.insert(s).second) {
                    throw ValidationError("Element " + elem_path + " (\"" + s +
                                          "\") is a duplicate within " + inner_path + ".");
                }

                indices.push_back(find_sample_index(y_samples, s));
            }

            result.push_back(std::move(indices));
        }

        return result;
    }

    // Gets a 16-bit ID for the specified key, which
    // corresponds to a key in y. Returns 0xFFFF if specified
    // key does not exist in y.
    std::uint16_t configuration::id(std::size_t _y_key) const {

        std::size_t lo =0;
        std::size_t hi =y.size();

        while (lo < hi) {
            std::size_t mid = lo + (hi - lo) / 2;
            if (y[mid].hash == _y_key) {
                return static_cast<std::uint16_t>(mid);
            }
            if (y[mid].hash < _y_key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return 0xffff;
    }

    // Boolean type-cast operator returns true iff the configuration
    // was loaded successfully by the constructor.
    configuration::operator bool() const {
        return okay;
    }

    // Constructor loads the features configuration from 
    // the specified istream object. Reads a JSON document from the istream, 
    // parses it with nlohmann/json and validates it against the following structure:
    //
    //   {
    //     "x":    { "<key>": [non-negative integers, at least one], ... },
    //     "y":    { "<key>": [non-negative integers, at least one], ... },
    //     "args": [ ["<key in y>", ...], ... ]
    //   }
    //
    // Extra root-level keys are allowed and ignored.
    //
    // Integers must lie in [0, INT_MAX] and must not repeat within one array.
    // "x" and "y" are stored as std::vector<Sample> sorted by key hash (the key
    // string itself is discarded); each Sample's integers are sorted ascending.
    // Hash collisions between keys of the same object are rejected. "args" is
    // stored as std::vector<std::vector<int>> holding indices into the sorted
    // "y" vector. The stored data is not used further.

    configuration::configuration(std::istream& _is) {

        const std::string content(std::istreambuf_iterator<char>(_is),
                           std::istreambuf_iterator<char>());

        json root;

        try {
            if (_is.bad()) {
                throw std::runtime_error("I/O error while reading file.");
            }

            root = json::parse(content);

            validate_root(root);

            const json& x = root.at("x");
            const json& y = root.at("y");
            const json& args = root.at("args");

            config::x=validate_integer_map(x, "x");
            config::y=validate_integer_map(y, "y");

            validate_key_separation(x, y);

            config::args=validate_args(args, x, y, config::y);

            okay=true;

        } catch (const json::parse_error& e) {
            std::cerr << "Error: malformed JSON: " << e.what() << '\n';
        } catch (const ValidationError& e) {
            std::cerr << "Error: invalid input: " << e.what() << '\n';
        } catch (const json::exception& e) {
            std::cerr << "Error: JSON processing failure: " << e.what() << '\n';
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << '\n';
        }
    }

} // namespace PhantomX