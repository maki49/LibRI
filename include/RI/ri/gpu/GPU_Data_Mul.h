// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"

namespace RI
{

namespace GPU_Data
{

template<typename TA, typename TAC, typename Tdata_CPU>
class Mul : public Pair_Packs<TA,TAC>
{
    Thread_Batch<const Pack*> pending_first;
    Thread_Batch<Pack> pending_second;
  public:
    GPU_Backend::Queue queue_ = nullptr;
	using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;

	const Pack &insert_1st(const TA &Aa, const TAC &Ab, const Shape_Vector &shape)
	{
		Pack &pack = this->local_packs()[Aa][Ab];
		if(!pack.exist)
		{
			pack.exist = true;
			pack.shape = shape;
		}
		pending_first.push_back(&pack);
		return pack;
	}

    void prepare_first()
    {
        if(!pending_first.size()) return;
        this->merge_packs();
        pending_first.for_each([&](const Pack *p){h_array_1.push_back(*p);});
        pending_first.release(); this->release_local_packs();
    }
    void prepare_second()
    {
        pending_second.for_each([&](const Pack &p){h_array_2.push_back(p);});
        pending_second.release();
    }
	void upload_1st(GPU_Backend::Queue queue)
	{
        this->queue_=queue;
        prepare_first();
		GPU_Backend::allocate(&this->d_data, this->totalSize, queue);
		GPU_Backend::memset(
			this->d_data, 0, this->totalSize * sizeof(Tdata_GPU), queue);

		const std::size_t batchCount = this->h_array_1.size();
		host_pointers_1.resize(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			host_pointers_1[i] = this->d_data + this->h_array_1[i].pos;
		GPU_Backend::allocate(&this->d_array_1, batchCount, queue);
		GPU_Backend::upload(
			batchCount, host_pointers_1.data(), this->d_array_1, queue);
	}

	const Pack &find_2nd(const TA &Aa, const TAC &Ab)
	{
		return Global_Func::find(this->ptrList, Aa, Ab);
	}

	const Pack &insert_2nd(const Pack &pack)
	{
		pending_second.push_back(pack);
		return pack;
	}

	// 在 GPU 上分配空间，并获取指针
	void upload_2nd(GPU_Backend::Queue queue)
	{
        this->queue_=queue;
        prepare_second();
		const std::size_t batchCount = this->h_array_2.size();
		host_pointers_2.resize(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			host_pointers_2[i] = this->d_data + this->h_array_2[i].pos;
		GPU_Backend::allocate(&this->d_array_2, batchCount, queue);
		GPU_Backend::upload(
			batchCount, host_pointers_2.data(), this->d_array_2, queue);
	}

	~Mul()
	{
		GPU_Backend::free(this->d_data, this->queue_);
		GPU_Backend::free(this->d_array_1, this->queue_);
		GPU_Backend::free(this->d_array_2, this->queue_);
	}

	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	HostMemoryResource host_storage{HOST_BUFFER_MAPPING_THRESHOLD};
    std::pmr::vector<Tdata_GPU*> host_pointers_1{&host_storage},host_pointers_2{&host_storage};
    std::pmr::vector<Pack> h_array_1{&host_storage};				// 记录每个batch的Pack
	std::pmr::vector<Pack> h_array_2{&host_storage};				// 记录每个batch的Pack
	Tdata_GPU **d_array_1 = nullptr;				// 记录每个batch的 d_data 指针（GPU）
	Tdata_GPU **d_array_2 = nullptr;				// 记录每个batch的 d_data 指针（GPU）
};

}

}

#endif
