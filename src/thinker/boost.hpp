#ifndef BOOST_THINKER_HPP
#define BOOST_THINKER_HPP

#include <boost.hpp>

namespace PhantomX {

    // The boost thinker is responsible for making predictions using a trained boost model.
    struct boost::thinker {
        struct result {

        };
        // Constructor initializes the thinker with a trained boost model
        // and the configuration/grouping of the attributes and features.
        thinker(const boost& _model, const config& _config);
        // Predicts the target value for the given sample using the trained boost model.
        result predict(const indexes& _sample) const;
    private:
        const boost& model_; // reference to the trained boost model.
        const config& config_; // reference to the configuration/grouping of the attributes and features.
    };

} // namespace PhantomX

#endif // BOOST_THINKER_HPP