// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"
#include <stdexcept>

namespace RI
{

namespace GPU_Data
{

template<typename TA, typename TAC, typename Tdata_CPU>
class Tmp
{
    Thread_Batch<std::size_t> pending;
  public:
    GPU_Backend::Queue queue_ = nullptr;
    using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;

    // Shapes are used immediately by the caller; only the element count persists.
    Pack insert(const Shape_Vector &shape)
    {
        Pack pack; pack.exist=true; pack.shape=shape;
        pending.push_back(shape.get_shape_all()); return pack;
    }
    void prepare_layout()
    {
        pending.for_each([&](std::size_t size){totalSize+=size;h_array.push_back(size);});
        pending.release();
    }
    std::vector<GPU_Backend::Int> upload(std::size_t memory_limit,GPU_Backend::Queue queue)
    {
        this->queue_=queue;
        prepare_layout();
        const std::size_t limit=std::min(totalSize,memory_limit/sizeof(Tdata_GPU));
        for(auto size:h_array)
            if(size>limit) throw std::length_error("LibRI temporary tensor exceeds segment capacity");
        GPU_Backend::allocate(&d_data,limit, queue);
        GPU_Backend::memset(d_data,0,limit*sizeof(Tdata_GPU),queue);

        host_pointers.reserve(h_array.size());
        std::vector<GPU_Backend::Int> segments;
        std::size_t used=0;
        GPU_Backend::Int count=0;
        for(auto size:h_array)
        {
            if(count && size>limit-used) { segments.push_back(count); used=0; count=0; }
            host_pointers.push_back(d_data+used);
            used+=size; ++count;
        }
        if(count) segments.push_back(count);
        GPU_Backend::allocate(&d_array,host_pointers.size(), queue);
        GPU_Backend::upload(host_pointers.size(),host_pointers.data(),d_array,queue);
        return segments;
    }

	~Tmp()
	{
		GPU_Backend::free(this->d_data, this->queue_);
		GPU_Backend::free(this->d_array, this->queue_);
	}

	std::size_t totalSize = 0;					// 总的数据数量
	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	HostMemoryResource host_storage{HOST_BUFFER_MAPPING_THRESHOLD};
    std::pmr::vector<Tdata_GPU*> host_pointers{&host_storage};
    std::pmr::vector<std::size_t> h_array{&host_storage};					// 记录每个batch的Pack
	Tdata_GPU **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）
};

}

}

#endif
