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
    // One descriptor per unique input block; contributions store only its index.
    // Atom keys resolve geometry-resident addresses; slot/offset resolve uploaded values.
    struct Block
    {
        TA atom;
        TAC partner;
        std::size_t offset;
        int owner;
    };
	struct Layout
	{
		std::map<TA, std::map<TAC, Pack>> ptr_list;
		std::vector<Block> blocks;
        std::vector<std::size_t> block_ids;
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
        std::vector<std::unordered_map<std::size_t,std::size_t>> ids_by_slot(h_data.size());
        for(const auto &row:this->ptrList) for(const auto &block:row.second)
        {
            const Pack &pack=block.second;
            const std::size_t id=unique_blocks.size();
            unique_blocks.push_back({row.first,block.first,pack.pos,pack.thread_num});
            ids_by_slot[pack.thread_num].emplace(pack.pos,id);
        }
        pending.for_each([&](const Entry *entry){
            batch_block_ids.push_back(ids_by_slot[entry->thread_num].at(entry->pos));
        });
        pending.release(); this->release_local_packs(); layout_ready=true;
    }

    // Transfer newly built metadata to the cache; numerical copy tasks stay local.
    Layout take_layout()
    {
        prepare_layout();
        return {std::move(this->ptrList),std::move(unique_blocks),std::move(batch_block_ids),std::move(planned_sizes)};
    }
    void bind_layout(const Layout &layout)
    {
        cached_layout = &layout;
        this->bind_pairs(layout.ptr_list);
        layout_ready = true;
    }
    const std::vector<Block>& blocks() const { return cached_layout ? cached_layout->blocks : unique_blocks; }
    const std::vector<std::size_t>& block_ids() const { return cached_layout ? cached_layout->block_ids : batch_block_ids; }
    const std::vector<std::size_t>& slot_sizes() const { return cached_layout ? cached_layout->sizes : planned_sizes; }

	void load_layout(const Layout &layout, const Matrix_Map &source)
	{
        bind_layout(layout);
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
			buffer.resize(slot_sizes()[tid]);
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
        // Bind each unique block once; all repeated GEMM references use dense IDs.
        std::vector<Tdata_GPU*> block_pointers(blocks().size());
        if(this->resident_blocks)
        {
            for(std::size_t id=0;id<blocks().size();++id)
            {
                const Block &block=blocks()[id];
                block_pointers[id]=resident_blocks->at(block.atom).at(block.partner);
            }
        }
        else
        {
            std::vector<std::size_t> slot_begin(h_data.size()+1,0);
            for(std::size_t i=0;i<h_data.size();++i) slot_begin[i+1]=slot_begin[i]+h_data[i].size();
            GPU_Backend::allocate(&d_data,slot_begin.back(),queue);
            for(std::size_t i=0;i<h_data.size();++i)
                GPU_Backend::upload(h_data[i].size(),h_data[i].data(),d_data+slot_begin[i],queue);
            for(std::size_t id=0;id<blocks().size();++id)
            {
                const Block &block=blocks()[id];
                block_pointers[id]=d_data+slot_begin[block.owner]+block.offset;
            }
        }
        std::vector<Tdata_GPU*> batch_pointers(block_ids().size());
        for(std::size_t i=0;i<block_ids().size();++i)
            batch_pointers[i]=block_pointers[block_ids()[i]];
        GPU_Backend::allocate(&d_array,batch_pointers.size(),queue);
        GPU_Backend::upload(batch_pointers.size(),batch_pointers.data(),d_array,queue);
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
    const Layout *cached_layout = nullptr;

  public:
	std::vector<std::vector<Tdata_GPU>> h_data;		// 存储数据（CPU）
	Tdata_GPU *d_data = nullptr;					// 存储数据（GPU）
	Tdata_GPU **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）
    std::vector<Block> unique_blocks;
    std::vector<std::size_t> batch_block_ids;
};

}

}

#endif
