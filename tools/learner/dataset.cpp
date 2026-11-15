#include <algorithm>
#include <iostream>
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

        // Parse the whole document; 
        // trailing content is rejected by the parser.
        json root;

        if (_is.bad()) {
            std::cerr << "Error: failed while reading file dataset\n";
            goto END;
        }

        try {
            root = json::parse(content);
        } catch (const json::parse_error& e) {
            std::cerr << "Error: " << std::string("malformed JSON: ") + e.what() << "\n";
            goto END;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << std::string("failed to parse JSON: ") + e.what() << "\n";
            goto END;
        }

        // Root validation.
        if (!root.is_array()) {
            std::cerr << "Error: root JSON value must be an array\n";
            goto END;
        }
        if (root.empty()) {
            std::cerr << "Error: root array must not be empty\n";
            goto END;
        }

        samples.reserve(root.size());

        for (std::size_t i = 0; i < root.size(); ++i) {

            const json& entry = root[i];
            const std::string ERROR = "Error: entry " + std::to_string(i);

            if (!entry.is_array()) {
                std::cerr << ERROR << ": entry must be an array\n";
                goto END;
            }
            if (entry.size() != 2) {
                std::cerr << ERROR << ": entry must contain exactly 2 elements, found " << entry.size() << "\n";
                goto END;
            }
            if (!entry[0].is_array()) {
                std::cerr << ERROR << ", element 0: must be an array of strings\n";
                goto END;
            }
            if (!entry[1].is_array()) {
                std::cerr << ERROR << ", element 1: must be an array of integers\n";
                goto END;
            }

            const json& y = entry[0];
            const json& x = entry[1];

            if (y.empty()) {
                std::cerr << ERROR << ", element 0: string array must not be empty\n";
                goto END;
            }
            if (y.size() != k) {
                std::cerr << ERROR << ", element 0: string array must contain exactly " << k << " strings, found " << y.size() << "\n";
                goto END;
            }

            sample sample;

            for (std::size_t j =0; j < y.size(); ++j) {
                if (!y[j].is_string()) {
                    std::cerr << ERROR << ", string index " << j << ": element must be a string\n";
                    goto END;
                }
                sample.y[j]=_config.id(y[j].get<std::string>());
            }

            sample.x.reserve(x.size());

            for (std::size_t j = 0; j < x.size(); ++j) {
                int value = 0;
                const std::string problem = read_non_negative_int(x[j], value);
                if (!problem.empty()) {
                    std::cerr << ERROR << ", integer index " << j << ": " << problem << "\n";
                    goto END;
                }
                sample.x.push_back(value);
            }
            std::sort(sample.x.begin(), sample.x.end());

            for (std::size_t j = 1; j < x.size(); j++) {
                if (sample.x[j - 1] == sample.x[j]) {
                    std::cerr << ERROR << ", duplicate values.\n";
                    goto END;
                }
            }
            samples.push_back(std::move(sample));
        }
        okay=true;
    END:
    }

} // namespace PhantomX