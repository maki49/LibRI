// ===================
//  Author: Peize Lin
//  date: 2022.07.23
// ===================

#pragma once

#include "Parallel_LRI.h"
#include "../comm/mix/Communicate_Tensors_Map_Judge.h"
#include "../global/Shared_Tensor_Map.h"

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
class Parallel_LRI_Equally: public Parallel_LRI<TA,Tcell,Ndim,Tdata>
{
  public:
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;
	using Tatom_pos = std::array<double,Ndim>;		// tmp

	virtual void set_parallel(
		const MPI_Comm &mpi_comm_in,
		const std::map<TA,Tatom_pos> &atoms_pos,
		const std::array<Tatom_pos,Ndim> &latvec,
		const std::array<Tcell,Ndim> &period_in,
		const std::set<Label::Aab_Aab> &labels) override;

//	std::map<TA,std::map<TAC,Tensor<Tdata>>> comm_tensors_map2(
//		const Label::ab &label,
//		const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds) const override;
	virtual std::map<TA,std::map<TAC,Tensor<Tdata>>> comm_tensors_map2(
		const std::vector<Label::ab> &label,
		const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds) const override;

	using Tensor_Judge = Communicate_Tensors_Map_Judge::Combined_Judge<TA, TC>;
	Tensor_Judge tensor_judge(const std::vector<Label::ab> &labels) const;

	// Collective global exchange, receiving the exact union of local requests
	// only on each node's root. The computation partition itself is unchanged.
	std::map<TA,std::map<TAC,Tensor<Tdata>>> comm_tensors_map2_node(
		const std::vector<Label::ab> &labels,
		const std::map<TA,std::map<TAC,Tensor<Tdata>>> &values,
		const Node_Shared::Communicator &node) const
	{
		const auto judge = Node_Shared::collect_on_root(node, tensor_judge(labels),
			[](Tensor_Judge &all, const Tensor_Judge &part)
			{
				auto &a = std::get<0>(all.judge_list); const auto &b = std::get<0>(part.judge_list);
				a.insert(a.end(), b.begin(), b.end());
				auto &c = std::get<1>(all.judge_list); const auto &d = std::get<1>(part.judge_list);
				c.insert(c.end(), d.begin(), d.end());
			});
		return Communicate_Tensors_Map::comm_map2(this->mpi_comm, values, judge);
	}

	const std::vector<TA >& get_list_Aa01() const override { return this->list_Aa01; }
	const std::vector<TAC>& get_list_Aa2 (const TA &Aa01) const override { return this->list_Aa2;  }
	const std::vector<TAC>& get_list_Ab01(const TA &Aa01, const TAC &Aa2) const override { return this->list_Ab01; }
	const std::vector<TAC>& get_list_Ab2 (const TA &Aa01, const TAC &Aa2, const TAC &Ab01) const override { return this->list_Ab2;  }

  public:	// private:
	std::vector<TA>  list_Aa01;
	std::vector<TAC> list_Aa2;
	std::vector<TAC> list_Ab01;
	std::vector<TAC> list_Ab2;

	MPI_Comm mpi_comm;
	std::array<Tcell,Ndim> period;

  public:	// private:
	virtual void set_parallel_loop4(
		const std::vector<TA> &atoms_vec);
	virtual void set_parallel_loop3(
		const std::vector<TA> &atoms_vec,
		const std::set<Label::Aab_Aab> &labels);
	virtual void set_atoms_loop3(
		const std::vector<TA> &atoms_vec,
		const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,TC>>>> &atoms_split_list1,
		const std::vector<std::vector<std::pair<TA,TC>>> &atoms_split_list2,
		const std::set<Label::Aab_Aab> &labels);
};

}

#include "Parallel_LRI_Equally.hpp"