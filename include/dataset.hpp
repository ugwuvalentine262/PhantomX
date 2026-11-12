#ifndef DATASET_HPP
#define DATASET_HPP

#include <string>
#include <vector>
#include <cstdint>

#include "indexes.hpp"

namespace PhantomX {

struct dataset {

    struct label {
        uint16_t id_;
        uint16_t args_[7];
    };

    struct sample {
        indexes x;
        label y;
    };

    std::vector<sample> samples_;

    groups groups_;

public:

    size_t size() const;

    bool load(const std::string& _filename);

    void clear();
};

} // namespace PhantomX

#endif // DATASET_HPP