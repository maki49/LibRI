#ifdef NDEBUG
#undef NDEBUG
#endif
#include <RI/ri/gpu/GPU_Data_Pack.h>
#include <cassert>
#include <complex>
#include <limits>
#include <iostream>
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

int main() { check_input_buffer(); std::cout << "PASS host input lifetime and exception cleanup\n"; }
