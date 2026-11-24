#ifndef DATASET_HPP
#define DATASET_HPP

#include <istream>
#include <vector>
#include <cstdint>

#include <configuration.hpp>
#include <indexes.hpp>

namespace PhantomX {

struct dataset {

    struct sample {
        indexes x; // attributes
        uint16_t y[8]={0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff}; // target
    };

    // the samples that make-up the dataset.
    std::vector<sample> samples;
    // this would be set to true if the dataset
    // is loaded successfully.
    bool okay=false;

public:

    // Boolean type-cast operator returns true iff the dataset
    // was loaded successfully by the constructor.
    operator bool() const;
    // Index operator returns a reference to the sample
    // at the specified index.
    const sample& operator[](std::size_t _index) const;
    // Gets the number of samples in 
    // the dataset.
    size_t size() const;
    // Constructor loads the dataset from the specified
    // input-stream object, wrt the specified configuration.
    dataset(std::istream _is, const config& _config);
    // Deleted default construction.
    dataset()=delete;

};

} // namespace PhantomX

#endif // DATASET_HPP