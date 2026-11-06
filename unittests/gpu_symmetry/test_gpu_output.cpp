#include "tensor_test_helpers.h"
#include "RI/ri/gpu/GPU_Data_Output.h"

using namespace tensor_test;

template <typename T>
void check_output(RI::GPU_Backend::Queue queue, T imaginary)
{
    RI::GPU_Data::Output<int, AtomCell, T> output;
    Tensors<T> expected;
    // Deliberately register in a different order from the output map.
    const std::vector<std::pair<int, AtomCell>> keys{
        {1, {0, {2}}}, {0, {1, {0}}}, {1, {1, {1}}}, {0, {0, {2}}}};
    const std::vector<RI::Shape_Vector> shapes{{3, 5}, {2, 7}, {1, 4}, {4, 3}};
    for (std::size_t i = 0; i < keys.size(); ++i)
    {
        output.insert(keys[i].first, keys[i].second, shapes[i]);
        // Multiple GEMMs may contribute to one physical output block.
        output.insert(keys[i].first, keys[i].second, shapes[i]);
        expected[keys[i].first][keys[i].second] =
            make_tensor(shapes[i], keys[i].first, keys[i].second.first, 2, int(i), imaginary);
    }
    output.upload(queue);
    using Gpu = RI::GPU_Backend::Type_to_GPU<T>;
    std::vector<Gpu> values(output.totalSize);
    for (const auto &row : output.ptrList)
        for (const auto &entry : row.second)
        {
            const auto &tensor = expected.at(row.first).at(entry.first);
            for (std::size_t i = 0; i < tensor.data->size(); ++i)
                values[entry.second.pos + i] = (*tensor.data)[i];
        }
    RI::GPU_Backend::upload(values.size(), values.data(), output.d_data, queue);
    RI::GPU_Backend::sync(queue);
    // Exercise a block larger than the budget, several chunks, and one chunk.
    for (const std::size_t bytes : {std::size_t(1), std::size_t(256), std::size_t(4096)})
    {
        Tensors<T> actual;
        output.download(actual, queue, T(1), bytes);
        compare(actual, expected, "chunked output initialization");
        double sum = 1;
        for (const double factor : {-1.0, 0.0, 0.25, 1.5})
        {
            output.download(actual, queue, T(factor), bytes);
            sum += factor;
            auto scaled = copy_tensors(expected);
            for (auto &row : scaled)
                for (auto &entry : row.second)
                    *entry.second.data *= T(sum);
            compare(actual, scaled, "chunked output accumulation");
        }
    }
    RI::GPU_Data::Output<int, AtomCell, T> empty;
    auto actual = copy_tensors(expected);
    empty.download(actual, queue, T(1), 1);
    compare(actual, expected, "empty output preserves destination");
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    try
    {
        RI::GPU_Backend::Context context(MPI_COMM_WORLD);
        check_output<double>(context.queue(), 0.0);
        check_output<std::complex<double>>(context.queue(), {0.0, 1.0});
        std::cout << "PASS bounded GPU output downloads\n";
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
