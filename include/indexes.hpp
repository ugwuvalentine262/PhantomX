#ifndef INDEXES_HPP
#define INDEXES_HPP

#include <vector>

namespace PhantomX {

using indices = std::vector<int>;

class indexes {
    indices data_;
public:
    operator bool() const;
    size_t size() const;
    int operator[](size_t _index) const;
    indexes operator&(const indexes& _indexes) const;
    indexes(indices _data={});
};

using group = indexes;
using groups = std::vector<group>;

} // namespace PhantomX

#endif // INDEXES_HPP