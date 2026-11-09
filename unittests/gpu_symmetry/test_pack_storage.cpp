#ifdef NDEBUG
#undef NDEBUG
#endif
#include <RI/ri/gpu/GPU_Data_Pack.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <complex>
#include <limits>
#include <iostream>
#include <vector>

struct Record : RI::GPU_Data::Pack
{
    int left = 0, right = 0;
};
struct Pairs : RI::GPU_Data::Pair_Packs<int, int, Record>
{
    using RI::GPU_Data::Pair_Packs<int, int, Record>::local_packs;
    using RI::GPU_Data::Pair_Packs<int, int, Record>::merge_packs;
    using RI::GPU_Data::Pair_Packs<int, int, Record>::release_local_packs;
};

// Construction failure must release both partially constructed objects and storage.
struct Tracked_Input
{
    static int live;
    static int remaining;
    int value = 17;
    Tracked_Input()
    {
        if (remaining-- == 0) throw std::runtime_error("test construction failure");
        ++live;
    }
    ~Tracked_Input() { --live; }
};
int Tracked_Input::live = 0;
int Tracked_Input::remaining = 0;

struct Counting_Resource : std::pmr::memory_resource
{
    RI::GPU_Data::HostMemoryResource upstream;
    int allocations = 0;
    void* do_allocate(std::size_t bytes, std::size_t alignment) override
    {
        void* result = upstream.allocate(bytes, alignment);
        ++allocations;
        return result;
    }
    void do_deallocate(void* memory, std::size_t bytes, std::size_t alignment) override
    {
        upstream.deallocate(memory, bytes, alignment);
        --allocations;
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
    {
        return this == &other;
    }
};

void check_input_buffer()
{
    using RI::GPU_Data::Default_Initialized_Buffer;
    Counting_Resource resource;
    {
        Default_Initialized_Buffer<double> empty(&resource, 0);
        assert(empty.size() == 0 && empty.data() == nullptr && resource.allocations == 0);
        Default_Initialized_Buffer<double> values(&resource, 32768);
        // Write all values before reading; default-initialized doubles are indeterminate.
#pragma omp parallel for
        for (int i = 0; i < 32768; ++i) values.data()[i] = i + 0.25;
        for (int i = 0; i < 32768; ++i) assert(values.data()[i] == i + 0.25);
        Default_Initialized_Buffer<std::complex<double>> complex_values(&resource, 7);
        for (int i = 0; i < 7; ++i) assert(complex_values.data()[i] == std::complex<double>(0));
        Tracked_Input::remaining = 7;
        Default_Initialized_Buffer<Tracked_Input> objects(&resource, 7);
        assert(Tracked_Input::live == 7);
        for (int i = 0; i < 7; ++i) assert(objects.data()[i].value == 17);
    }
    assert(resource.allocations == 0 && Tracked_Input::live == 0);
    Tracked_Input::remaining = 2;
    bool threw = false;
    try { Default_Initialized_Buffer<Tracked_Input> objects(&resource, 7); }
    catch (const std::runtime_error&) { threw = true; }
    assert(threw && resource.allocations == 0 && Tracked_Input::live == 0);
    threw = false;
    try { Default_Initialized_Buffer<double> overflow(&resource, std::numeric_limits<std::size_t>::max()); }
    catch (const std::length_error&) { threw = true; }
    assert(threw && resource.allocations == 0);
}

int main()
{
    check_input_buffer();
    RI::GPU_Data::HostMemoryResource resource;
    RI::GPU_Data::HostMemoryResource buffer_resource{64 * 1024};
    for (auto* current : {&resource, &buffer_resource})
    for (const std::size_t alignment : {std::size_t(16), std::size_t(8192)})
        for (const std::size_t bytes : {std::size_t(0), std::size_t(257), std::size_t(131329)})
        {
            void* memory = current->allocate(bytes, alignment);
            assert(memory && reinterpret_cast<std::uintptr_t>(memory) % alignment == 0);
            if (bytes) std::memset(memory, 1, bytes);
            current->deallocate(memory, bytes, alignment);
        }
    omp_set_dynamic(0);
    for (int threads : {1, 4, 16})
    {
        omp_set_num_threads(threads);
        Pairs pairs;
        RI::GPU_Data::Thread_Batch<Record*> records;
        RI::GPU_Data::Thread_Batch<int> dimensions;
        constexpr int count = 200003;
#pragma omp parallel for schedule(dynamic, 17)
        for (int i = 0; i < count; ++i)
        {
            const int left = i % 3, right = i % 7;
            auto& record = pairs.local_packs()[left][right];
            if (!record.exist)
            {
                record.exist = true;
                record.left = left;
                record.right = right;
                record.shape = {std::size_t(right + 1), 2};
            }
            records.push_back(&record);
            dimensions.push_back(right + 1);
        }
        int unique = 0;
        pairs.merge_packs([&](const Record&, const RI::GPU_Data::Pack&) { ++unique; });
        assert(unique == 21 && pairs.totalSize == 168);
        assert(records.size() == count && dimensions.size() == count);
        std::vector<int> expected;
        records.for_each([&](const Record* record) {
            const auto& global = pairs.ptrList.at(record->left).at(record->right);
            assert(record->pos == global.pos);
            assert(record->pos + record->shape.get_shape_all() <= pairs.totalSize);
            expected.push_back(record->right + 1);
        });
        std::size_t i = 0;
        dimensions.for_each([&](int value) { assert(value == expected[i++]); });
        records.release();
        dimensions.release();
        pairs.release_local_packs();
        for (const auto& left : pairs.ptrList)
            for (const auto& right : left.second)
                assert(right.second.shape.get_shape_all() == std::size_t(2 * (right.first + 1)));
        assert(records.size() == count && dimensions.size() == count);
        std::cout << "PACK_STORAGE_PASS threads=" << threads << " entries=" << count
                  << " unique=" << unique << '\n';
    }
}
