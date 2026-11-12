#ifndef DECISION_TREE
#define DECISION_TREE

#include <string>
#include <cstdint>

#include "indexes.hpp"

namespace PhantomX {

struct decision_tree {

    struct output {};

    struct node { int left, right, index, group; };

    struct leaf {
        struct label { uint16_t id_, args_[7]; int occurrence; };
        std::vector<label> labels;
        size_t id;
    };

    using nodes = std::vector<node>;
    using leafs = std::vector<leaf>;

private:

    groups groups_;
    nodes nodes_;
    leafs leafs_;

public:

    bool save(const std::string& _filename) const;

    bool load(const std::string& _filename);

    void clear();

    void set(int _left, int _right, int _index, int _group);

    output operator()(const indexes& _attributes) const;

    decision_tree(groups _groups, nodes _nodes, leafs _leafs);
    decision_tree(const decision_tree& _dt);
    decision_tree();

};

} // namespace PhantomX

#endif // DECISION_TREE