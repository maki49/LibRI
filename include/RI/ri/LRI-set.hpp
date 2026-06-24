// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "LRI.h"
#include "RI_Tools.h"
#include "Label_Tools.h"
#include "../global/Map_Operator.h"
#include "../global/MPI_Wrapper-func.h"
#include "../global/Cereal_Types.h"
#include "../parallel/Parallel_LRI_Equally.h"
#include <cereal/archives/binary.hpp>
#include <algorithm>
#include <sstream>

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::set_parallel(
	const MPI_Comm &mpi_comm_in,
	const std::map<TA,Tatom_pos> &atoms_pos,
	const std::array<Tatom_pos,Ndim> &latvec,
	const std::array<Tcell,Ndim> &period_in,
	const std::vector<Label::ab_ab> &labels_all)
{
	this->mpi_comm = mpi_comm_in;
	this->period = period_in;
	this->parallel->set_parallel(
		this->mpi_comm, atoms_pos, latvec, this->period,
		Label_Tools::to_Aab_Aab_set(labels_all));
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::set_tensors_map2(
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_local,
	const std::vector<Label::ab> &label_list,
	const std::map<std::string, double> &para_in,
	const std::string &save_name_in)
{
	const std::map<std::string, double> para_default = {
		{"flag_period",      true},
		{"flag_comm",        (MPI_Wrapper::mpi_get_size(this->mpi_comm)>1)
		                     ? true : false},
		{"flag_filter",      true},
		{"threshold_filter", 0.0}};
	const std::map<std::string, double> para = Map_Operator::cover(para_default, para_in);

	std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_new =
		para.at("flag_period")
		? RI_Tools::cal_period(Ds_local, this->period)
		: Ds_local;

	if(para.at("flag_comm"))
		Ds_new = this->parallel->comm_tensors_map2(label_list, std::move(Ds_new));

	if(para.at("flag_filter"))
	{
		std::vector<RI_Tools::T_filter_func<Tdata>> filter_func_list;
		for(const Label::ab &label : label_list)
			filter_func_list.push_back(this->filter_funcs[label]);
		Ds_new = RI_Tools::filter(std::move(Ds_new), filter_func_list, para.at("threshold_filter"));
	}

	const std::string save_name =
		save_name_in!="default"
		? save_name_in
		: Label_Tools::get_name(label_list);
	for(const Label::ab &label : label_list)
		this->data_ab_name[label] = save_name;

	// --- Intra-node shared memory: node root writes to MPI_Win, others map it read-only ---
	{
		auto *par_eq = dynamic_cast<Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>*>(this->parallel.get());
		if(para.at("flag_comm") && par_eq && par_eq->node_comm.flag_allocate
		   && MPI_Wrapper::mpi_get_size(par_eq->node_comm()) > 1)
		{
			const MPI_Comm &nc = par_eq->node_comm();
			const bool is_node_root = (MPI_Wrapper::mpi_get_rank(nc) == 0);

			// Build layout: (A, AC) -> (window offset, tensor shape)
			std::map<TA, std::map<TAC, std::pair<std::size_t, Shape_Vector>>> layout_map;
			std::size_t n_local = 0;
			if(is_node_root)
			{
				for(const auto &kv_a : Ds_new)
					for(const auto &kv_ac : kv_a.second)
					{
						layout_map[kv_a.first][kv_ac.first] = {n_local, kv_ac.second.shape};
						n_local += kv_ac.second.shape.get_shape_all();
					}
			}

			// Collective: allocate shared window (root owns all elements; others own 0)
			auto pool = std::make_shared<SharedMemoryPool<Tdata>>();
			pool->allocate(nc, n_local);

			// Root copies Ds_new data into its window segment
			if(is_node_root)
			{
				Tdata *wp = pool->local_ptr();
				for(const auto &kv_a : Ds_new)
					for(const auto &kv_ac : kv_a.second)
					{
						const std::size_t off = layout_map.at(kv_a.first).at(kv_ac.first).first;
						const std::size_t n   = kv_ac.second.shape.get_shape_all();
						std::copy(kv_ac.second.ptr(), kv_ac.second.ptr() + n, wp + off);
					}
			}

			// Fence: root write visible to all on-node processes
			pool->sync();

			// Broadcast layout_map from root (cereal binary + MPI_Bcast)
			{
				std::string buf;
				if(is_node_root)
				{
					std::ostringstream oss;
					{ cereal::BinaryOutputArchive ar(oss); ar(layout_map); }
					buf = oss.str();
				}
				int buf_sz = static_cast<int>(buf.size());
				MPI_Bcast(&buf_sz, 1, MPI_INT, 0, nc);
				if(!is_node_root) buf.resize(buf_sz);
				if(buf_sz > 0)	MPI_Bcast(&buf[0], buf_sz, MPI_BYTE, 0, nc);
				if(!is_node_root)
				{
					std::istringstream iss(buf);
					cereal::BinaryInputArchive ar(iss);
					ar(layout_map);
				}
			}

			// All processes construct non-owning Tensors pointing into root's window
			Tdata *root_ptr = pool->query_ptr(0);
			const std::shared_ptr<void> anchor = pool;	// make Tensor take part in Pool's reference count
			std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_shm;
			for(const auto &kv_a : layout_map)
				for(const auto &kv_ac : kv_a.second)
					Ds_shm[kv_a.first][kv_ac.first] = Tensor<Tdata>(
						kv_ac.second.second, root_ptr + kv_ac.second.first, anchor);

			Ds_new = std::move(Ds_shm);
			this->data_pool[save_name].shm_pool = pool;
		}
	}

	this->data_pool[save_name].Ds_ab = std::move(Ds_new);

	this->data_pool[save_name].index_Ds_ab = RI_Tools::get_index(this->data_pool[save_name].Ds_ab);
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::free_tensors_map2(
	const std::string &save_name)
{
	this->data_pool.erase(save_name);
}

}