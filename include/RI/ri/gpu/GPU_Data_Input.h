// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"

#include <algorithm>
#include <omp.h>
#include <type_traits>
#include <unordered_map>
#include <stdexcept>

namespace RI
{

namespace GPU_Data
{

template<typename T> struct Input_Pack : Pack { const Tensor<T>* tensor=nullptr; };

template<typename TA, typename TAC, typename Tdata_CPU>
class Input : public Pair_Packs<TA,TAC,Input_Pack<Tdata_CPU>>
{
    using Entry=Input_Pack<Tdata_CPU>;
    Thread_Batch<const Entry*> pending;
    bool layout_ready=false, values_ready=false;
  public:
    GPU_Backend::Queue queue_ = nullptr;
	using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;
	using Same_data_layout = std::integral_constant<
		bool,
		sizeof(Tdata_CPU)==sizeof(Tdata_GPU) && alignof(Tdata_CPU)==alignof(Tdata_GPU)>;

	using Matrix_Map = std::map<TA, std::map<TAC, Tensor<Tdata_CPU>>>;
	using Block_Map = std::map<TA, std::map<TAC, Tdata_GPU*>>;

	// Numerical values are deliberately absent: a cached layout is valid only
	// while the caller's block-key and shape signature is unchanged.
	struct Layout
	{
		std::map<TA, std::map<TAC, Pack>> ptr_list;
		std::vector<Pack> array;
		std::vector<std::size_t> sizes;
	};

    // Resolve worker metadata to independent values before saving the layout.
    void prepare_layout()
    {
        if(layout_ready) return;
        this->merge_packs([&](const Entry &local,Pack &shared) {
            const std::size_t owner=next_owner++ % h_data.size();
            shared.thread_num=owner; shared.pos=planned_sizes[owner];
            planned_sizes[owner]+=shared.shape.get_shape_all();
            if(!resident_blocks) copy_tasks[owner].push_back({local.tensor,shared.pos});
        });
        pending.for_each([&](const Entry *entry){ h_array.push_back(*entry); });
        pending.release(); this->release_local_packs(); layout_ready=true;
    }
    Layout save_layout()
    {
        prepare_layout();
        return {this->ptrList,h_array,planned_sizes};
    }

	void load_layout(const Layout &layout, const Matrix_Map &source)
	{
		this->ptrList = layout.ptr_list;
		this->h_array = layout.array;
		this->planned_sizes = layout.sizes;
		this->h_data.resize(layout.sizes.size());
		this->copy_tasks.clear();
		this->copy_tasks.resize(layout.sizes.size());
		this->layout_ready = true;
		if(!this->resident_blocks)
			for(const auto &row : layout.ptr_list)
				for(const auto &block : row.second)
				{
					const Pack &pack = block.second;
					const Tensor<Tdata_CPU> &tensor = source.at(row.first).at(block.first);
					this->copy_tasks.at(pack.thread_num).push_back({&tensor, pack.pos});
				}
	}

	Input()
	{
		this->h_data.resize(omp_get_max_threads());
		this->planned_sizes.assign(h_data.size(),0);
        this->copy_tasks.resize(h_data.size());
	}

    // All registrations are deferred; the API remains usable by existing callers.
    void defer_input_copies() {}
    void bind_resident_blocks(const Block_Map *blocks) { resident_blocks=blocks; }

	void finish_deferred_copies()
	{
        prepare_layout();
        if(values_ready) return;
        values_ready=true;
		if(this->resident_blocks)
		{
			this->copy_tasks.clear();
			return;
		}
		#pragma omp parallel for schedule(static)
		for(int tid=0; tid<static_cast<int>(this->h_data.size()); ++tid)
		{
			std::vector<Tdata_GPU> &buffer = this->h_data[tid];
			buffer.resize(this->planned_sizes[tid]);
			for(const Copy_Task &task : this->copy_tasks[tid])
				copy_data_impl(buffer.data() + task.pos, *task.tensor, Same_data_layout());
		}
		this->copy_tasks.clear();
	}

    const Pack& insert(const TA &Aa,const TAC &Ab,const Tensor<Tdata_CPU> &tensor)
    {
        Entry &pack=this->local_packs()[Aa][Ab];
        if(!pack.exist) { pack.exist=true; pack.shape=tensor.shape; pack.tensor=&tensor; }
        pending.push_back(&pack);
        return pack;
    }
    void insert_data(const Tensor<Tdata_CPU>&) {}

	// 将对应的 C, V, D 放在 d_Cs, d_Vs, d_Ds 上
	void upload(GPU_Backend::Queue queue)
	{
        this->queue_=queue;
        finish_deferred_copies();
		if(this->resident_blocks)
		{
			// Resolve symbolic (thread, offset) locations to geometry-fixed device blocks.
			std::vector<std::unordered_map<std::size_t, Tdata_GPU*>> pointers(this->h_data.size());
			for(const auto &row : this->ptrList)
				for(const auto &block : row.second)
				{
					const Pack &pack = block.second;
					pointers.at(pack.thread_num)[pack.pos] = this->resident_blocks->at(row.first).at(block.first);
				}
			std::vector<Tdata_GPU*> batch_pointers(this->h_array.size());
			for(std::size_t i = 0; i < this->h_array.size(); ++i)
			{
				const Pack &pack = this->h_array[i];
				batch_pointers[i] = pointers.at(pack.thread_num).at(pack.pos);
			}
			GPU_Backend::allocate(&this->d_array, batch_pointers.size(), queue);
			GPU_Backend::upload(batch_pointers.size(), batch_pointers.data(), this->d_array, queue);
			return;
		}
		std::vector<std::size_t> h_data_begin(this->h_data.size()+1, 0);
		for(std::size_t i=1; i<h_data_begin.size(); ++i)
			h_data_begin[i] = h_data_begin[i-1] + this->h_data[i-1].size();

		GPU_Backend::allocate(&this->d_data, h_data_begin.back(), queue);
		for(std::size_t i=0; i<this->h_data.size(); ++i)
			GPU_Backend::upload(
				this->h_data[i].size(), this->h_data[i].data(),
				this->d_data+h_data_begin[i], queue);

		const std::size_t batchCount = this->h_array.size();
		std::vector<Tdata_GPU*> d_array_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for(std::size_t i=0; i<batchCount; ++i)
			d_array_[i] = this->d_data + this->h_array[i].pos + h_data_begin[this->h_array[i].thread_num];
		GPU_Backend::allocate(&this->d_array, batchCount, queue);
		GPU_Backend::upload(batchCount, d_array_.data(), this->d_array, queue);
	}

	~Input()
	{
		GPU_Backend::free(this->d_data, this->queue_);
		GPU_Backend::free(this->d_array, this->queue_);
	}

  private:
	struct Copy_Task
	{
		const Tensor<Tdata_CPU> *tensor;
		std::size_t pos;
	};

	static void copy_data_impl(Tdata_GPU *dest, const Tensor<Tdata_CPU> &tensor, std::true_type)
	{
		std::copy(tensor.ptr(), tensor.ptr() + tensor.shape.get_shape_all(), dest);
	}

	static void copy_data_impl(Tdata_GPU *dest, const Tensor<Tdata_CPU> &tensor, std::false_type)
	{
		const Tdata_CPU *src = tensor.ptr();
		for(std::size_t i=0; i<tensor.shape.get_shape_all(); ++i)
			dest[i] = GPU_Backend::data_to_GPU(src[i]);
	}

	std::size_t next_owner = 0;
	std::vector<std::size_t> planned_sizes;
	std::vector<std::vector<Copy_Task>> copy_tasks;
	const Block_Map *resident_blocks = nullptr;

  public:
	std::vector<std::vector<Tdata_GPU>> h_data;		// 存储数据（CPU）
	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	Tdata_GPU **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）
	std::vector<Pack> h_array;					// 记录每个batch的Pack
};

}

}

#endif
