#include "tensor_test_helpers.h"
#include "RI/global/Shared_Tensor_Map.h"
#include <sstream>

using namespace tensor_test;

template<typename T> void equal_tensor(const RI::Tensor<T> &a, const RI::Tensor<T> &b)
{
    if(a.shape.get_shape_all() != b.shape.get_shape_all()) throw std::runtime_error("shared tensor shape mismatch");
    for(std::size_t i = 0; i < a.shape.get_shape_all(); ++i)
        if(std::abs(a.ptr()[i] - b.ptr()[i]) > 1e-10 * (1 + std::abs(b.ptr()[i])))
            throw std::runtime_error("shared tensor value mismatch");
}

struct Select_Block
{
    int rank, size;
    bool judge(const std::tuple<long long, AtomCell> &key) const
    { return std::get<0>(key) == 5000000000LL || rank == size - 1; }
};

template<typename T> RI::Tensor<T> check_views(T imaginary)
{
    RI::Node_Shared::Communicator node(MPI_COMM_WORLD);
    using Map = std::map<long long, std::map<AtomCell, RI::Tensor<T>>>;
    Map root;
    auto reference = make_tensor({2,3}, 0, 1, 2, 3, imaginary);
    if(node.rank == 0)
    {
        root[5000000000LL][{0,{0}}] = reference;
        root[5000000001LL][{1,{0}}] = make_tensor({3,2}, 1, 0, 2, 4, imaginary);
    }
    auto empty = RI::Node_Shared::share_map(node, Map{}, Select_Block{node.rank,node.size});
    if(!empty.empty()) throw std::runtime_error("empty shared map changed");
    bool failed = false;
    try { RI::Node_Shared::Mapping::create(node, 4096, [](void*) { throw std::runtime_error("writer"); }); }
    catch(const std::runtime_error&) { failed = true; }
    if(!failed) throw std::runtime_error("writer failure did not reach all ranks");
    auto shared = RI::Node_Shared::share_map(node, std::move(root), Select_Block{node.rank,node.size});
    auto view = shared.at(5000000000LL).at({0,{0}});
    if(!view.is_external() || view.data) throw std::runtime_error("expected shared external storage");
    if(shared.size() != std::size_t(node.rank == node.size - 1 ? 2 : 1))
        throw std::runtime_error("rank-local block selection changed");
    equal_tensor(view, reference);
    equal_tensor(view.copy(), reference);
    equal_tensor(view.reshape({3,2}), reference.reshape({3,2}));
    equal_tensor(view.transpose(), reference.transpose());
    equal_tensor(view.conjugate(), reference.conjugate());
    equal_tensor(-view, -reference);
    equal_tensor(view + reference, reference + reference);
    equal_tensor(reference - view, reference - reference);
    equal_tensor(T(0.5)*view, T(0.5)*reference);
    equal_tensor(view*T(0.5), reference*T(0.5));
    for(double p : {1.0,2.0,3.0,std::numeric_limits<double>::max()})
        if(std::abs(view.norm(p)-reference.norm(p))>1e-10) throw std::runtime_error("shared norm mismatch");
    std::stringstream stream;
    { cereal::BinaryOutputArchive archive(stream); archive(view); }
    RI::Tensor<T> restored;
    { cereal::BinaryInputArchive archive(stream); archive(restored); }
    if(restored.is_external() || !restored.data) throw std::runtime_error("serialized view must load as owned storage");
    equal_tensor(restored, reference);
    // Only rank 0 writes; MAP_PRIVATE must protect other processes' pages.
    if(node.rank == 0) { view += reference; view -= reference; view.ptr()[0] += T(9); }
    MPI_Barrier(node.comm);
    auto expected = reference.copy();
    if(node.rank == 0) expected.ptr()[0] += T(9);
    equal_tensor(view, expected);
    // The view remains usable after all original maps/temporary communicators die.
    shared.clear();
    return view;
}

#ifdef __GPU_RI
using Input = RI::Label::ab;
template<typename T> using Lri = RI::LRI<int,int,1,T>;

template<typename T> void put(Lri<T> &lri, Input input, int version, T imaginary)
{
    int rank=0,size=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    Tensors<T> values;
    const bool c = input==Input::a || input==Input::b;
    // A pure sender on the last rank and empty input maps elsewhere exercise
    // receiving on a node root that did not originally own the tensors.
    if(rank==size-1)
        for(int a=0;a<2;++a) for(int b=0;b<2;++b) for(int cell=0;cell<3;++cell)
            values[a][{b,{cell}}]=make_tensor(c?RI::Shape_Vector{2,2,2}:RI::Shape_Vector{2,2},a,b,cell,version,imaginary);
    lri.set_tensors_map2(values,{input});
}

template<typename T> void verify(Lri<T> &plain, Lri<T> &shared, const std::vector<Label> &labels, bool gpu)
{
    plain.cal_mode=RI::LRI_Cal_Mode::CPU;
    Tensors<T> expected; plain.cal_loop3(labels,expected);
    shared.cal_mode=gpu?RI::LRI_Cal_Mode::GPU:RI::LRI_Cal_Mode::CPU;
    Tensors<T> actual; shared.cal_loop3(labels,actual);
    compare(actual,expected,"node-shared contraction");
    shared.cal_loop3(labels,actual,T(-0.5));
    for(auto &row:expected) for(auto &entry:row.second) *entry.second.data *= T(0.5);
    compare(actual,expected,"node-shared signed accumulation");
}

template<typename T> void check_lri(T imaginary, bool gpu, bool resident)
{
    const std::vector<Label> labels(RI::Label::array_ab_ab.begin(),RI::Label::array_ab_ab.end());
    Lri<T> plain,shared;
    for(auto *lri:{&plain,&shared})
        lri->set_parallel(MPI_COMM_WORLD,{{0,{0.0}},{1,{0.4}}},{{{{1.0}}}},Cell{{3}},labels);
    shared.set_gpu_shared_data_names({"a","b","a0b0"});
    if(resident) shared.set_gpu_resident_data_names({"a","b","a0b0"});
    for(auto input:RI::Label::array_ab) { put(plain,input,0,imaginary); put(shared,input,0,imaginary); }
    int ranks=0,rank=0; MPI_Comm_size(MPI_COMM_WORLD,&ranks); MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    const auto &a=shared.data_pool.at("a").Ds_ab;
    if(a.size()!=plain.data_pool.at("a").Ds_ab.size()) throw std::runtime_error("shared input keys differ");
    for(const auto &row:a) for(const auto &entry:row.second)
        if(entry.second.is_external()!=(ranks>1)) throw std::runtime_error("sharing/single-rank fallback was not exercised");
    for(auto label:labels) verify(plain,shared,{label},gpu);
    verify(plain,shared,labels,gpu);
    std::unique_ptr<Lri<T>> rank_zero_copy;
    RI::Tensor<T> retained,old_value;
    if(rank==0)
    {
        rank_zero_copy.reset(new Lri<T>(shared));
        retained=shared.data_pool.at("a").Ds_ab.begin()->second.begin()->second;
        old_value=retained.copy();
    }
    shared.free_tensors_map2("a");
    put(plain,Input::a,19,imaginary); put(shared,Input::a,19,imaginary);
    verify(plain,shared,labels,gpu);
    if(rank==0) equal_tensor(retained,old_value);
    // Local replacement with communication disabled must remain noncollective.
    if(rank==0)
    {
        Tensors<T> local; local[0][{0,{0}}]=make_tensor({2,2,2},0,0,0,5,imaginary);
        rank_zero_copy->set_tensors_map2(local,{Input::a},{{"flag_comm",0},{"flag_period",0}});
        if(rank_zero_copy->data_pool.at("a").Ds_ab.begin()->second.begin()->second.is_external())
            throw std::runtime_error("flag_comm=false did not retain private storage");
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
#endif

int main(int argc,char **argv)
{
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
    RI::Tensor<double> survivor;
    try
    {
        if(provided<MPI_THREAD_MULTIPLE) throw std::runtime_error("MPI_THREAD_MULTIPLE required");
        survivor=check_views<double>(0.0);
        check_views<std::complex<double>>({0.0,1.0});
#ifdef __GPU_RI
        const bool gpu=argc>1 && std::string(argv[1])=="gpu";
        for(bool resident:{false,true})
        {
            check_lri<double>(0.0,gpu,resident);
            check_lri<std::complex<double>>({0.0,1.0},gpu,resident);
        }
#endif
        std::cout<<"PASS shared tensor values, rank-local writes, lifetime, and contractions\n";
    }
    catch(const std::exception &error)
    { std::cerr<<error.what()<<'\n'; MPI_Abort(MPI_COMM_WORLD,1); return 1; }
    MPI_Finalize();
    // Destruction of the last view after MPI finalization must not invoke MPI.
    survivor=RI::Tensor<double>{};
    return 0;
}
