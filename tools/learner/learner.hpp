#include <string>
#include <span>

#include <dataset.hpp>
#include <decision_tree.hpp>

namespace PhantomX {

class learner {

public:

    struct result {
        int max_depth_=0;
        int success_=0;
        uint64_t elapsed_=0;
        decision_tree dt_;
    public:
        void write_json(const std::string& _filename);
        result(const decision_tree& _dt, int _max_depth, uint64_t _elapsed);
        result();
        
    };

private:

    dataset dataset_;

    int max_features_;
    int min_samples_leaf_;
    int min_samples_split_;

public:

    result train(const std::string& _dataset_file);

    learner(int _max_features=-1, int _min_samples_leaf=-1, int _min_samples_split=-1);
};

} // namespace PhantomX