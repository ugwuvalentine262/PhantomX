#include <algorithm>
#include <iostream>
#include <functional>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <dataset.hpp>

using json = nlohmann::json;

namespace PhantomX {

    // Validates a JSON value as a non-negative integer fitting in int.
    // Returns an empty string on success, otherwise an error description.
    static std::string read_non_negative_int(const json& value, int& out) {
        if (value.is_boolean()) {
            return "boolean is not an integer";
        }
        if (value.is_number_float()) {
            return "floating-point number is not allowed";
        }
        if (!value.is_number_integer()) { // is_number_integer covers signed and unsigned
            return "value is not an integer";
        }

        std::uint64_t magnitude = 0;
        if (value.is_number_unsigned()) {
            magnitude = value.get<std::uint64_t>();
        } else {
            const std::int64_t signed_value = value.get<std::int64_t>();
            if (signed_value < 0) {
                return "negative integers are not allowed";
            }
            magnitude = static_cast<std::uint64_t>(signed_value);
        }

        if (magnitude > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            return "integer is outside the supported int range";
        }
        out = static_cast<int>(magnitude);
        return std::string();
    }

    // Boolean type-cast operator returns true iff the dataset
    // was loaded successfully by the constructor.
    dataset::operator bool() const {
        return dataset::okay;
    }

    // Index operator returns a reference to the sample
    // at the specified index.
    const dataset::sample& dataset::operator[](std::size_t _index) const {
        return dataset::samples.at(_index);
    }

    // Gets the number of samples in 
    // the dataset.
    size_t dataset::size() const {
        return dataset::samples.size();
    }

    // Constructor loads the dataset from the specified
    // input-stream object, wrt the specified configuration.
    dataset::dataset(std::istream _is, const config& _config) {

        const std::size_t k =_config.args.size();

        const std::string content((std::istreambuf_iterator<char>(_is)),
                                  std::istreambuf_iterator<char>());

        json root;

        try {
            if (!_config) {
                throw ValidationError("Error: specified configuration is not defined.");
            }
            if (_is.bad()) {
                throw std::runtime_error("failed while reading dataset from istream.");
            }

            root = json::parse(content);
        
            // Root validation.
            if (!root.is_array()) {
                throw ValidationError("Error: root JSON value must be an array.");
            }
            if (root.empty()) {
                throw ValidationError("Error: root array must not be empty.");
            }

            samples.reserve(root.size());

            for (std::size_t i = 0; i < root.size(); ++i) {

                const json& entry = root[i];
                const std::string where = "entry " + std::to_string(i);

                if (!entry.is_array()) {
                    throw ValidationError(where + ": entry must be an array.");
                }
                if (entry.size() != 2) {
                    throw ValidationError(where + ": entry must contain exactly 2 elements, found " 
                                            + std::to_string(entry.size()) + ".");
                }
                if (!entry[0].is_array()) {
                    throw ValidationError(where + ", element 0: must be an array of strings.");
                }
                if (!entry[1].is_array()) {
                    throw ValidationError(where + ", element 1: must be an array of integers.");
                }

                const json& y = entry[0];
                const json& x = entry[1];

                if (y.empty()) {
                    throw ValidationError(where + ", element 0: string array must not be empty.");
                }
                if (y.size() != k) {
                    throw ValidationError(where + ", element 0: string array must contain exactly " + 
                                            std::to_string(k) + " strings, found " + 
                                            std::to_string(y.size()) + ".");
                }

                sample sample;

                for (std::size_t j =0; j < y.size(); ++j) {
                    if (!y[j].is_string()) {
                        throw ValidationError(where + ", string index " + std::to_string(j) + ": element must be a string.");
                    }
                    sample.y[j]=_config.id(
                                    std::hash<std::string>{}(y[j].get<std::string>())
                                 );
                }

                sample.x.reserve(x.size());

                for (std::size_t j = 0; j < x.size(); ++j) {
                    int value = 0;
                    const std::string problem = read_non_negative_int(x[j], value);
                    if (!problem.empty()) {
                        throw ValidationError(where + ", integer index " + std::to_string(j) + ": " + problem + ".");
                    }
                    sample.x.push_back(value);
                }
                std::sort(sample.x.begin(), sample.x.end());

                for (std::size_t j = 1; j < x.size(); j++) {
                    if (sample.x[j - 1] == sample.x[j]) {
                        throw ValidationError(where + ", duplicate values.");
                    }
                }
                samples.push_back(std::move(sample));
            }
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