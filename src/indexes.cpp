#include <iterator>
#include <algorithm>

#include <indexes.hpp>

namespace PhantomX {

    indexes::operator bool() const
    {
        return data_.empty()!=true;
    }

    size_t indexes::size() const
    {
        return data_.size();
    }

    int indexes::operator[](size_t _index) const
    {
        return data_.at(_index);
    }

    indexes indexes::operator&(const indexes& _indexes) const
    {
        indices intersect;

        std::set_intersection(
            data_.begin(), data_.end(),
            _indexes.data_.begin(), _indexes.data_.end(),
            std::back_inserter(intersect)
        );

        return intersect;
    }

    indexes::indexes(indices _data)
        :   data_(std::move(_data))
    {
        std::sort(data_.begin(), data_.end());
    }

} // namespace PhantomX