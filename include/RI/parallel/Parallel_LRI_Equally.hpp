// ===================
//  Author: Peize Lin
//  date: 2022.07.23
// ===================

#pragma once

#include "Parallel_LRI_Equally.h"
#include "../global/Global_Func-1.h"
#include "../global/MPI_Wrapper.h"
#include "../distribute/Distribute_Equally.h"
#include "../comm/mix/Communicate_Tensors_Map_Judge.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#define MPI_CHECK(x) if((x)!=MPI_SUCCESS)	throw std::runtime_error(std::string(__FILE__)+" line "+std::to_string(__LINE__));

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>::set_parallel(
	const MPI_Comm &mpi_comm_in,
	const std::map<TA,Tatom_pos> &atoms_pos,
	const std::array<Tatom_pos,Ndim> &latvec,
	const std::array<Tcell,Ndim> &period_in,
	const std::set<Label::Aab_Aab> &labels)
{
	this->mpi_comm = mpi_comm_in;
	this->period = period_in;
	const std::vector<TA> atoms_vec = Global_Func::map_key_to_vec(atoms_pos);

	this->set_parallel_loop4(atoms_vec);
	this->set_parallel_loop3(atoms_vec, labels);

	// --- Create intra-node communicator (MPI_COMM_TYPE_SHARED) ---
	{
		MPI_Wrapper::mpi_comm nc;
		MPI_CHECK( MPI_Comm_split_type(
			mpi_comm_in, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nc.comm) );
		nc.flag_allocate = true;
		this->node_comm = std::move(nc);
	}
	const MPI_Comm &nc = this->node_comm();
	const int node_size = MPI_Wrapper::mpi_get_size(nc);

	// --- Compute node-union atom lists via intra-node allgather + sort+unique ---

	// Allgather vector<TA> across nc, return sorted+unique union
	auto gather_TA = [&nc, node_size](const std::vector<TA>& local) -> std::vector<TA>
	{
		const int local_n = static_cast<int>(local.size());
		std::vector<int> counts(node_size), displs(node_size, 0);
		MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, nc);
		for(int i = 1; i < node_size; ++i)
			displs[i] = displs[i-1] + counts[i-1];
		const int total = displs[node_size-1] + counts[node_size-1];
		std::vector<TA> all(total);
		const MPI_Datatype dt = MPI_Wrapper::mpi_get_datatype(TA{});
		MPI_Allgatherv(local.empty() ? nullptr : local.data(), local_n, dt,
		               all.empty()  ? nullptr : all.data(),   counts.data(), displs.data(), dt, nc);
		// remove duplicates (union)
		std::sort(all.begin(), all.end());
		all.erase(std::unique(all.begin(), all.end()), all.end());
		return all;
	};

	// Allgather vector<TAC> across nc: flatten each TAC to (1+Ndim) ints, allgather,
	// reconstruct, sort+unique.  Assumes TA and Tcell are int-width integer types.
	auto gather_TAC = [&nc, node_size](const std::vector<TAC>& local) -> std::vector<TAC>
	{
		constexpr std::size_t STRIDE = 1 + Ndim;
		std::vector<int> flat;
		flat.reserve(local.size() * STRIDE);
		for(const auto& tac : local)
		{
			flat.push_back(static_cast<int>(tac.first));
			for(std::size_t d = 0; d < Ndim; ++d)
				flat.push_back(static_cast<int>(tac.second[d]));
		}
		const int local_n = static_cast<int>(flat.size());
		std::vector<int> counts(node_size), displs(node_size, 0);
		MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, nc);
		for(int i = 1; i < node_size; ++i)
			displs[i] = displs[i-1] + counts[i-1];
		const int total = displs[node_size-1] + counts[node_size-1];
		std::vector<int> all_flat(total);
		MPI_Allgatherv(flat.empty()    ? nullptr : flat.data(),    local_n, MPI_INT,
		               all_flat.empty()? nullptr : all_flat.data(), counts.data(), displs.data(), MPI_INT, nc);
		std::vector<TAC> all;
		all.reserve(static_cast<std::size_t>(total) / STRIDE);
		for(std::size_t i = 0; i + STRIDE <= static_cast<std::size_t>(total); i += STRIDE)
		{
			TA a = static_cast<TA>(all_flat[i]);
			TC c;
			for(std::size_t d = 0; d < Ndim; ++d)
				c[d] = static_cast<Tcell>(all_flat[i + 1 + d]);
			all.emplace_back(a, c);
		}
		std::sort(all.begin(), all.end());
		all.erase(std::unique(all.begin(), all.end()), all.end());
		return all;
	};

	this->list_Aa01_node = gather_TA(this->list_Aa01);
	this->list_Aa2_node  = gather_TAC(this->list_Aa2);
	this->list_Ab01_node = gather_TAC(this->list_Ab01);
	this->list_Ab2_node  = gather_TAC(this->list_Ab2);

	this->list_A_node.clear();
	for(const auto &kv : this->list_A)
	{
		const Label::Aab_Aab &lbl = kv.first;
		const List_A<TA,TAC> &la = kv.second;
		List_A<TA,TAC> &la_node = this->list_A_node[lbl];
		la_node.a01 = gather_TA(la.a01);
		la_node.a2  = gather_TAC(la.a2);
		la_node.b01 = gather_TAC(la.b01);
		la_node.b2  = gather_TAC(la.b2);
	}
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>::set_parallel_loop4(
	const std::vector<TA> &atoms_vec)
{
	constexpr std::size_t num_index = 4;

	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,TC>>>>
		atoms_split_list = Distribute_Equally::distribute_atoms_periods(
			this->mpi_comm, atoms_vec, this->period, num_index, false);

	this->list_Aa01 = atoms_split_list.first;
	this->list_Aa2  = atoms_split_list.second[0];
	this->list_Ab01 = atoms_split_list.second[1];
	this->list_Ab2  = atoms_split_list.second[2];
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>::set_parallel_loop3(
	const std::vector<TA> &atoms_vec,
	const std::set<Label::Aab_Aab> &labels)
{
	constexpr std::size_t num_index = 2;
	const std::vector<TAC> atoms_period_vec = Divide_Atoms::traversal_atom_period(atoms_vec, this->period);

	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,TC>>>>
		atoms_split_list1 = Distribute_Equally::distribute_atoms_periods(
			this->mpi_comm, atoms_vec, this->period, num_index, false);
	const std::vector<std::vector<std::pair<TA,TC>>>
		atoms_split_list2 = Distribute_Equally::distribute_periods(
			this->mpi_comm, atoms_vec, this->period, num_index, false);
	for(const Label::Aab_Aab &label : labels)
	{
		List_A<TA,TAC> &atoms = this->list_A[label];
		atoms.a01 = atoms_vec;
		atoms.a2 = atoms_period_vec;
		atoms.b01 = atoms_period_vec;
		atoms.b2 = atoms_period_vec;
		switch(label)
		{
			case Label::Aab_Aab::a01b01_a01b01:
				atoms.a2  = atoms_split_list2[0];
				atoms.b01 = atoms_split_list2[1];
				break;
			case Label::Aab_Aab::a01b01_a2b01:
				atoms.a01 = atoms_split_list1.first;
				atoms.b01 = atoms_split_list1.second[0];
				break;
			case Label::Aab_Aab::a01b01_a01b2:
				atoms.b01 = atoms_split_list1.second[0];
				atoms.a01 = atoms_split_list1.first;
				break;
			case Label::Aab_Aab::a01b01_a2b2:
				atoms.a01 = atoms_split_list1.first;
				atoms.b2  = atoms_split_list1.second[0];
				break;
			case Label::Aab_Aab::a01b2_a2b01:
				atoms.a01 = atoms_split_list1.first;
				atoms.b01 = atoms_split_list1.second[0];
				break;
			default:
				throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
		}
	}
}


/*
template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
auto Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>::comm_tensors_map2(
	const Label::ab &label,
	const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds) const
-> std::map<TA,std::map<TAC,Tensor<Tdata>>>
{
	switch(label)
	{
		case Label::ab::a:
			return Communicate_Tensors_Map_Judge::comm_map2(this->mpi_comm, Ds, Global_Func::to_set(this->list_Aa01), Global_Func::to_set(this->list_Aa2));
		case Label::ab::b:
			return Communicate_Tensors_Map_Judge::comm_map2_period(this->mpi_comm, Ds, Global_Func::to_set(this->list_Ab01), Global_Func::to_set(this->list_Ab2), this->period);
		case Label::ab::a0b0:	case Label::ab::a0b1:
		case Label::ab::a1b0:	case Label::ab::a1b1:
			return Communicate_Tensors_Map_Judge::comm_map2(this->mpi_comm, Ds, Global_Func::to_set(this->list_Aa01), Global_Func::to_set(this->list_Ab01));
		case Label::ab::a0b2:	case Label::ab::a1b2:
			return Communicate_Tensors_Map_Judge::comm_map2(this->mpi_comm, Ds, Global_Func::to_set(this->list_Aa01), Global_Func::to_set(this->list_Ab2));
		case Label::ab::a2b0:	case Label::ab::a2b1:
			return Communicate_Tensors_Map_Judge::comm_map2_period(this->mpi_comm, Ds, Global_Func::to_set(this->list_Aa2), Global_Func::to_set(this->list_Ab01), this->period);
		case Label::ab::a2b2:
			return Communicate_Tensors_Map_Judge::comm_map2_period(this->mpi_comm, Ds, Global_Func::to_set(this->list_Aa2), Global_Func::to_set(this->list_Ab2), this->period);
		default:
			throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
	}
}
*/

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
auto Parallel_LRI_Equally<TA,Tcell,Ndim,Tdata>::comm_tensors_map2(
	const std::vector<Label::ab> &label_list,
	const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds) const
-> std::map<TA,std::map<TAC,Tensor<Tdata>>>
{
	std::tuple<
		std::vector<std::tuple< std::set<TA>, std::set<std::pair<TA,TC>> >>,
		std::vector<std::tuple< std::set<std::pair<TA,TC>>, std::set<std::pair<TA,TC>> >>
		> s_list;

	std::vector<bool> flags(6, false);
	for(const Label::ab &label : label_list)
	{
		switch(label)
		{
			case Label::ab::a:
				flags[0]=true;	break;
			case Label::ab::b:
				flags[1]=true;	break;
			case Label::ab::a0b0:	case Label::ab::a0b1:
			case Label::ab::a1b0:	case Label::ab::a1b1:
				flags[2]=true;	break;
			case Label::ab::a0b2:	case Label::ab::a1b2:
				flags[3]=true;	break;
			case Label::ab::a2b0:	case Label::ab::a2b1:
				flags[4]=true;	break;
			case Label::ab::a2b2:
				flags[5]=true;	break;
			default:
				throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
		}
	}

	// Shared-memory mode: node root receives for the whole node (node-union judge);
	// non-root processes use empty judge and act as pure senders.
	const bool using_shm = this->node_comm.flag_allocate;
	const bool is_node_root = !using_shm ||
		(MPI_Wrapper::mpi_get_rank(this->node_comm()) == 0);

	if(is_node_root)
	{
		// used in loop3
		const auto &la_map = using_shm ? this->list_A_node : this->list_A;
		for(const auto &list_atom : la_map)
		{
			if(flags[0])	std::get<0>(s_list).push_back(std::make_tuple( Global_Func::to_set(list_atom.second.a01), Global_Func::to_set(list_atom.second.a2)  ));
			if(flags[1])	std::get<1>(s_list).push_back(std::make_tuple( Global_Func::to_set(list_atom.second.b01), Global_Func::to_set(list_atom.second.b2)  ));
			if(flags[2])	std::get<0>(s_list).push_back(std::make_tuple( Global_Func::to_set(list_atom.second.a01), Global_Func::to_set(list_atom.second.b01) ));
			if(flags[3])	std::get<0>(s_list).push_back(std::make_tuple( Global_Func::to_set(list_atom.second.a01), Global_Func::to_set(list_atom.second.b2)  ));
			if(flags[4])	std::get<1>(s_list).push_back(std::make_tuple( Global_Func::to_set(list_atom.second.a2),  Global_Func::to_set(list_atom.second.b01) ));
			if(flags[5])	std::get<1>(s_list).push_back(std::make_tuple( Global_Func::to_set(list_atom.second.a2),  Global_Func::to_set(list_atom.second.b2)  ));
		}

		// used in loop4
		const std::vector<TA>  &Aa01 = using_shm ? this->list_Aa01_node : this->list_Aa01;
		const std::vector<TAC> &Aa2  = using_shm ? this->list_Aa2_node  : this->list_Aa2;
		const std::vector<TAC> &Ab01 = using_shm ? this->list_Ab01_node : this->list_Ab01;
		const std::vector<TAC> &Ab2  = using_shm ? this->list_Ab2_node  : this->list_Ab2;

		if(flags[0])	std::get<0>(s_list).push_back(std::make_tuple( Global_Func::to_set(Aa01), Global_Func::to_set(Aa2) ));
		if(flags[1])	std::get<1>(s_list).push_back(std::make_tuple( Global_Func::to_set(Ab01), Global_Func::to_set(Ab2) ));
		if(flags[2])	std::get<0>(s_list).push_back(std::make_tuple( Global_Func::to_set(Aa01), Global_Func::to_set(Ab01) ));
		if(flags[3])	std::get<0>(s_list).push_back(std::make_tuple( Global_Func::to_set(Aa01), Global_Func::to_set(Ab2) ));
		if(flags[4])	std::get<1>(s_list).push_back(std::make_tuple( Global_Func::to_set(Aa2), Global_Func::to_set(Ab01) ));
		if(flags[5])	std::get<1>(s_list).push_back(std::make_tuple( Global_Func::to_set(Aa2), Global_Func::to_set(Ab2) ));
	}
	// else: non-root on a shared-memory node — s_list stays empty → empty judge → no receive

	return Communicate_Tensors_Map_Judge::comm_map2_combine_origin_period(this->mpi_comm, Ds, s_list, this->period);
}

}

#undef MPI_CHECK
