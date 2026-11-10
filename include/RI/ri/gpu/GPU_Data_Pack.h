// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once
#include <omp.h>
#include <map>
#include <memory_resource>
#include <new>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <vector>
#include "../../global/Shape_Vector.h"

#ifdef __GPU_RI

namespace RI
{

namespace GPU_Data
{

inline constexpr std::size_t HOST_BUFFER_MAPPING_THRESHOLD = 64 * 1024;

// Metadata chunks must return to the OS after packing, rather than remain
// resident in per-worker malloc arenas. Larger alignments and non-POSIX
// platforms use the standard resource.
class HostMemoryResource : public std::pmr::memory_resource
{
	const std::size_t minimum_mapping_size;

	bool use_mapping(std::size_t bytes, std::size_t alignment) const
	{
#if defined(__unix__) || defined(__APPLE__)
		static const long page_size = ::sysconf(_SC_PAGESIZE);
		return bytes >= minimum_mapping_size && page_size > 0 &&
			   (std::size_t(page_size) & (std::size_t(page_size) - 1)) == 0 &&
			   alignment <= std::size_t(page_size);
#else
		return false;
#endif
	}

	void* do_allocate(std::size_t bytes, std::size_t alignment) override
	{
#if defined(__unix__) || defined(__APPLE__)
		if (use_mapping(bytes, alignment))
		{
			void* memory = ::mmap(nullptr, bytes ? bytes : 1, PROT_READ | PROT_WRITE,
								  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
			if (memory == MAP_FAILED) throw std::bad_alloc();
			return memory;
		}
#endif
		return std::pmr::new_delete_resource()->allocate(bytes, alignment);
	}

	void do_deallocate(void* memory, std::size_t bytes, std::size_t alignment) override
	{
#if defined(__unix__) || defined(__APPLE__)
		if (use_mapping(bytes, alignment))
		{
			::munmap(memory, bytes ? bytes : 1);
			return;
		}
#endif
		std::pmr::new_delete_resource()->deallocate(memory, bytes, alignment);
	}

	bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
	{
		return this == &other;
	}

public:
	explicit HostMemoryResource(std::size_t minimum = 0) : minimum_mapping_size(minimum) {}
};

struct Pack
{
	bool exist = false;
	std::size_t pos = -1;
	Shape_Vector shape;
	int thread_num = 0;
};
// Related operands and dimensions append in the same worker-local order.
// Flat arrays own values; caches never retain pointers into these temporary chunks.
template<typename T>
class Thread_Batch
{
    struct alignas(64) Chunk
    {
        HostMemoryResource storage{HOST_BUFFER_MAPPING_THRESHOLD};
        std::pmr::vector<T> values{&storage};
    };
    std::vector<Chunk> chunks{static_cast<std::size_t>(omp_get_max_threads())};
public:
    void push_back(const T &value) { chunks[omp_get_thread_num()].values.push_back(value); }
    std::size_t size() const { std::size_t n=0; for(const auto &c:chunks) n+=c.values.size(); return n; }
    template<typename F> void for_each(F f) const { for(const auto &c:chunks) for(const auto &v:c.values) f(v); }
    void release() { for(auto &c:chunks) std::pmr::vector<T>(c.values.get_allocator()).swap(c.values); }
};

// Register unique pairs privately, then resolve one shared offset per block.
template<typename TA, typename TAC, typename Record=Pack>
class Pair_Packs
{
    // Each worker owns an arena; map nodes are destroyed before bulk release.
    using Local_Map=std::pmr::map<TA,std::pmr::map<TAC,Record>>;
    struct alignas(64) Shard
    {
        HostMemoryResource upstream;
        std::pmr::monotonic_buffer_resource storage{&upstream};
        Local_Map values{&storage};
    };
    std::vector<Shard> shards{static_cast<std::size_t>(omp_get_max_threads())};
protected:
    auto& local_packs() { return shards[omp_get_thread_num()].values; }
    template<typename F> void merge_packs(F first_insert)
    {
        for(auto &shard:shards) for(auto &row:shard.values) for(auto &block:row.second)
        {
            Record &local=block.second;
            Pack &shared=ptrList[row.first][block.first];
            if(!shared.exist)
            {
                shared=local; shared.pos=totalSize;
                totalSize+=shared.shape.get_shape_all();
                first_insert(local,shared);
            }
            static_cast<Pack&>(local)=shared;
        }
    }
    void merge_packs() { merge_packs([](const Record&,Pack&){}); }
    void release_local_packs() { for(auto &shard:shards) { shard.values.clear(); shard.storage.release(); } }
public:
    std::size_t totalSize=0;
    std::map<TA,std::map<TAC,Pack>> ptrList;
};


}

}

#endif
