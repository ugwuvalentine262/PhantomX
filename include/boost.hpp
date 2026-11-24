#ifndef BOOST_HPP
#define BOOST_HPP

#include <vector>

#include "decision_tree.hpp"

namespace PhantomX {

// An implementation of a gradient boosting algorithm for
// the machine learning model. This class is designed to be used in conjunction with
// other components of the PhantomX library to provide a complete machine learning solution.
struct boost {

    class learner; // forward declaration of the gradient boosting learning algorithm.
    class thinker; // forward declaration of the gradient boosting prediction algorithm.

private:
    std::uint64_t goal_id_; // the goal/target id of the boosting model.
    std::vector<decision_tree> trees_; // the decision-tree models that make up the gradient boosting model.

public:
    // Writes the nodes and leaf data of the gradient boosting model
    // to the specified output-stream. Returns true iff the
    // operation completed successfully.
    bool write(std::ostream& _os) const;
    // Reads the nodes and leaf data of the gradient boosting model
    // from the specified input-stream. Returns true iff the
    // operation completed successfully.
    bool read(std::istream& _is);

};

} // namespace PhantomX

#endif // BOOST_HPP