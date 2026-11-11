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
    Thread_Batch<std::size_t> pending_second;
    const std::vector<std::size_t> *cached_first = nullptr, *cached_second = nullptr;
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

    // Resolve repeated block references to offsets; retain full shapes only per unique pair.
    void prepare_first()
    {
        if(!pending_first.size()) return;
        this->merge_packs();
        pending_first.for_each([&](const Pack *p){first_batch_offsets.push_back(p->pos);});
        pending_first.release(); this->release_local_packs();
    }
    void prepare_second()
    {
        pending_second.for_each([&](std::size_t offset){second_batch_offsets.push_back(offset);});
        pending_second.release();
    }
    const std::vector<std::size_t>& first_offsets() const { return cached_first ? *cached_first : first_batch_offsets; }
    const std::vector<std::size_t>& second_offsets() const { return cached_second ? *cached_second : second_batch_offsets; }
    void bind_first(const typename Pair_Packs<TA,TAC>::Pack_Map &map, const std::vector<std::size_t> &array)
    { this->bind_pairs(map); cached_first = &array; }
    void bind_second(const std::vector<std::size_t> &array) { cached_second = &array; }
	void upload_1st(GPU_Backend::Queue queue)
	{
        this->queue_=queue;
        prepare_first();
		GPU_Backend::allocate(&this->d_data, this->totalSize, queue);
		GPU_Backend::memset(
			this->d_data, 0, this->totalSize * sizeof(Tdata_GPU), queue);

		const std::size_t batchCount = first_offsets().size();
		std::vector<Tdata_GPU*> d_array_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			d_array_[i] = this->d_data + first_offsets()[i];
		GPU_Backend::allocate(&this->d_array_1, batchCount, queue);
		GPU_Backend::upload(
			batchCount, d_array_.data(), this->d_array_1, queue);
	}

	const Pack &find_2nd(const TA &Aa, const TAC &Ab)
	{
		return Global_Func::find(this->pairs(), Aa, Ab);
	}

	const Pack &insert_2nd(const Pack &pack)
	{
		pending_second.push_back(pack.pos);
		return pack;
	}

	// 在 GPU 上分配空间，并获取指针
	void upload_2nd(GPU_Backend::Queue queue)
	{
        this->queue_=queue;
        prepare_second();
		const std::size_t batchCount = second_offsets().size();
		std::vector<Tdata_GPU*> d_array_2_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			d_array_2_[i] = this->d_data + second_offsets()[i];
		GPU_Backend::allocate(&this->d_array_2, batchCount, queue);
		GPU_Backend::upload(
			batchCount, d_array_2_.data(), this->d_array_2, queue);
	}

	~Mul()
	{
		GPU_Backend::free(this->d_data, this->queue_);
		GPU_Backend::free(this->d_array_1, this->queue_);
		GPU_Backend::free(this->d_array_2, this->queue_);
	}

	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	std::vector<std::size_t> first_batch_offsets;				// One element offset per GEMM contribution; shapes stay in the unique-pair map.
	std::vector<std::size_t> second_batch_offsets;				// One element offset per GEMM contribution; shapes stay in the unique-pair map.
	Tdata_GPU **d_array_1 = nullptr;				// 记录每个batch的 d_data 指针（GPU）
	Tdata_GPU **d_array_2 = nullptr;				// 记录每个batch的 d_data 指针（GPU）
};

}

}

#endif
