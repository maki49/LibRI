#include "tensor_test_helpers.h"

using namespace tensor_test;

template <typename T>
void check_tiling(T imaginary, bool resident)
{
    RI::LRI<int, int, 1, T> lri;
    const std::vector<Label> labels(RI::Label::array_ab_ab.begin(), RI::Label::array_ab_ab.end());
    lri.set_parallel(MPI_COMM_WORLD, {{0, {0.0}}, {1, {0.3}}, {2, {0.6}}}, {{{{1.0}}}}, Cell{{3}}, labels);
    int index = 0;
    for (const auto input : RI::Label::array_ab)
    {
        const bool coefficient = input == RI::Label::ab::a || input == RI::Label::ab::b;
        const RI::Shape_Vector shape = coefficient ? RI::Shape_Vector{2, 2, 2} : RI::Shape_Vector{2, 2};
        Tensors<T> tensors;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int cell = 0; cell < 3; ++cell)
                    tensors[a][{b, {cell}}] = make_tensor(shape, a, b, cell, index, imaginary);
        lri.set_tensors_map2(tensors, {input});
        ++index;
    }
    if (resident) lri.set_gpu_resident_data_names({"a", "b", "a0b0"});
    for (std::size_t index = 0; index <= labels.size(); ++index)
    {
        const std::vector<Label> selected = index == labels.size() ? labels : std::vector<Label>{labels[index]};
        lri.cal_mode = RI::LRI_Cal_Mode::CPU;
        Tensors<T> expected;
        lri.cal_loop3(selected, expected);
        auto actual = copy_tensors(expected);
        lri.cal_mode = RI::LRI_Cal_Mode::GPU;
        // Reuse a warm full plan, discard it for tiles, handle an uneven last
        // tile (3 atoms / 2), then rebuild the full plan on the same object.
        const std::vector<std::size_t> sizes{0, 0, 1, 2, 0};
        const std::vector<double> factors{0.25, -1.0, 0.5, 0.75, 0.0};
        double sum = 1;
        for (std::size_t step = 0; step < sizes.size(); ++step)
        {
            lri.gpu_atom_batch_size = sizes[step];
            lri.cal_loop3(selected, actual, T(factors[step]));
            sum += factors[step];
            auto scaled = copy_tensors(expected);
            for (auto &row : scaled)
                for (auto &entry : row.second)
                    *entry.second.data *= T(sum);
            compare(actual, scaled, "tiling label=" + std::to_string(index)
                + " batch=" + std::to_string(sizes[step]));
        }
    }
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    try
    {
        if (RI::MPI_Wrapper::mpi_get_size(MPI_COMM_WORLD) != 1)
            throw std::runtime_error("Tiling regression requires one MPI rank");
        for (const bool resident : {false, true})
        {
            check_tiling<double>(0.0, resident);
            check_tiling<std::complex<double>>({0.0, 1.0}, resident);
        }
        std::cout << "PASS tiled contractions and plan-cache transitions\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
