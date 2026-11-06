// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"
#include "../LRI_Cal_Aux.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace RI
{

namespace GPU_Data
{

template<typename TA, typename TAC, typename Tdata_CPU>
class Output
{
  public:
	using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;
	using Same_data_layout = std::integral_constant<
		bool,
		sizeof(Tdata_CPU)==sizeof(Tdata_GPU) && alignof(Tdata_CPU)==alignof(Tdata_GPU)>;

	const Pack &insert(const TA &Aa, const TAC &Ab, const Shape_Vector &shape)
	{
		Pack &pack = this->ptrList[Aa][Ab];
		if(!pack.exist)
		{
			pack.exist = true;
			pack.pos = this->totalSize;
			pack.shape = shape;
			this->totalSize += shape.get_shape_all();
		}
		this->h_array.push_back(pack);
		return pack;
	}

	void upload(GPU_Backend::Queue queue)
	{
		GPU_Backend::allocate(&this->d_data, this->totalSize);
		GPU_Backend::memset(
			this->d_data, 0, totalSize * sizeof(Tdata_GPU), queue);

		const std::size_t batchCount = this->h_array.size();
		std::vector<Tdata_GPU*> d_array_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			d_array_[i] = this->d_data + this->h_array[i].pos;
		GPU_Backend::allocate(&this->d_array, batchCount);
		GPU_Backend::upload(batchCount, d_array_.data(), this->d_array, queue);
	}

	void download(
		std::map<TA, std::map<TAC, Tensor<Tdata_CPU>>> &Ds_result,
		GPU_Backend::Queue queue,
		const Tdata_CPU fac) const
	{
		download(Ds_result, queue, fac, std::size_t(64) * 1024 * 1024);
	}

	// Download contiguous groups of complete output blocks. The staging buffer
	// is bounded by buffer_bytes, or by the largest block when that is larger.
	void download(
		std::map<TA, std::map<TAC, Tensor<Tdata_CPU>>> &Ds_result,
		GPU_Backend::Queue queue,
		const Tdata_CPU fac,
		const std::size_t buffer_bytes) const
	{
		if(buffer_bytes == 0)
			throw std::invalid_argument("GPU output download buffer must be positive");
		if(totalSize == 0)
			return;

		struct Block
		{
			const TA *a;
			const TAC *b;
			const Pack *pack;
		};
		std::vector<Block> blocks;
		std::size_t capacity = std::max(std::size_t(1), buffer_bytes / sizeof(Tdata_GPU));
		for(const auto &row : ptrList)
			for(const auto &entry : row.second)
			{
				blocks.push_back({&row.first, &entry.first, &entry.second});
				capacity = std::max(capacity, entry.second.shape.get_shape_all());
			}
		// Registration order need not match map order, particularly with OpenMP.
		std::sort(blocks.begin(), blocks.end(), [](const Block &a, const Block &b)
		{
			return a.pack->pos < b.pack->pos;
		});
		std::vector<Tdata_GPU> h_data(std::min(totalSize, capacity));
		for(std::size_t begin = 0; begin < blocks.size(); )
		{
			const std::size_t offset = blocks[begin].pack->pos;
			std::size_t end = begin;
			std::size_t count = 0;
			while(end < blocks.size())
			{
				const Pack &pack = *blocks[end].pack;
				const std::size_t next_count = pack.pos + pack.shape.get_shape_all() - offset;
				if(next_count > h_data.size()) break;
				count = next_count;
				++end;
			}
			GPU_Backend::download(count, d_data + offset, h_data.data(), queue);
			GPU_Backend::sync(queue);
			for(std::size_t i = begin; i < end; ++i)
			{
				const Block &block = blocks[i];
				const Pack &pack = *block.pack;
				Tensor<Tdata_CPU> D_output(pack.shape);
				const Tdata_GPU *source = h_data.data() + pack.pos - offset;
				download_data_impl(D_output, source, Same_data_layout());
				LRI_Cal_Aux::add_Ds(std::move(D_output), Ds_result[*block.a][*block.b], fac);
			}
			begin = end;
		}
	}

	~Output()
	{
		GPU_Backend::free(this->d_data);
		GPU_Backend::free(this->d_array);
	}

  private:
	static void download_data_impl(
		Tensor<Tdata_CPU> &D_output,
		const Tdata_GPU *source,
		std::true_type)
	{
		std::memcpy(
			D_output.ptr(),
			source,
			D_output.shape.get_shape_all()*sizeof(Tdata_CPU));
	}

	static void download_data_impl(
		Tensor<Tdata_CPU> &D_output,
		const Tdata_GPU *source,
		std::false_type)
	{
		const std::size_t size = D_output.shape.get_shape_all();
		const auto ptr_src = source;
		const auto ptr_dest = D_output.ptr();
		for(std::size_t i=0; i<size; ++i)
			ptr_dest[i] = GPU_Backend::data_to_CPU(ptr_src[i]);
	}

  public:
	std::size_t totalSize = 0;                  // 总的数据数量
	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	std::vector<Pack> h_array;					// 记录每个batch的Pack
	Tdata_GPU **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）
	std::map<TA, std::map<TAC, Pack>> ptrList;	// 记录每个原子对的Pack
};

}

}

#endif
