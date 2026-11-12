#include <decision_tree.hpp>

namespace PhantomX {

    /// saves the decision tree model to the file of the 
    /// specified name.
    /// @return true if the operation is successful
    /// otherwise false.
    bool decision_tree::save(const std::string& _filename) const
    {

    }

    /// loads the decision tree model from the file 
    /// of the specified name.
    /// @return true if the operation is successful, otherwise false.
    bool decision_tree::load(const std::string& _filename)
    {

    }

    /// @brief  clears the tree.
    /// tree will be null after this operation.
    void decision_tree::clear()
    {

    }

    /// @brief sets a node of the decision tree model.
    /// @param _feature the ID of the feature used for splitting.
    /// @param _node the index of the node in the array.
    /// @param _left the index of the left child.
    /// @param _right the index of the right child.
    void decision_tree::set(int _feature, int _node, int _left, int _right)
    {

    }

    /// @brief runs the prediction algorithm on the decision tree model.
    /// @param _features the input-features to be processed.
    /// @return the output of the decision tree model.
    decision_tree::output decision_tree::operator()(const features& _features) const
    {

    }

    /// @brief class constructor
    /// @param _node_reserve number of nodes to reserve to avoid continuous allocation.
    /// @param _leaf_reserve number of leafs to reserve to avoid continuous allocation.
    decision_tree::decision_tree(int _node_reserve, int _leaf_reserve)
        :   node_idx_(0)
        ,   leaf_idx_(0)
    {
        nodes_.reserve(_node_reserve);
        leafs_.reserve(_leaf_reserve);
    }

} // namespace PhantomX