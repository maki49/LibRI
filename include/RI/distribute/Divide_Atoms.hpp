// ===================
//  Author: Peize Lin
//  date: 2022.07.13
// ===================

#pragma once

#include "Divide_Atoms.h"
#include "../global/Global_Func-3.h"

#include <numeric>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <functional>
#include <queue>
#include <cassert>

namespace RI
{

namespace Divide_Atoms
{
	template<typename TA>
	std::size_t atom_weight(
		const std::map<TA,std::size_t> &weights,
		const TA &atom)
	{
		const auto ptr = weights.find(atom);
		return (ptr==weights.end() || ptr->second==0) ? 1 : ptr->second;
	}

	// Core distribution function. 
	// Assign the heaviest item to the currently lightest group.
	inline std::vector<std::size_t> assign_owners_weighted(
		const std::size_t group_size,
		const std::vector<std::size_t> &item_weights)
	{
		assert(group_size>0);

		std::vector<std::size_t> order(item_weights.size());
		std::iota(order.begin(), order.end(), std::size_t(0));
		std::stable_sort(order.begin(), order.end(),
			[&item_weights](const std::size_t i, const std::size_t j) -> bool
			{ return item_weights[i] > item_weights[j]; });

		using Tload_group = std::pair<std::size_t,std::size_t>;	// weight, atom-index
		std::priority_queue<Tload_group, std::vector<Tload_group>, std::greater<Tload_group>> groups;	//min heap
		for(std::size_t group=0; group<group_size; ++group)
			groups.push(std::make_pair(std::size_t(0), group));

		std::vector<std::size_t> owners(item_weights.size());
		for(const std::size_t i : order)
		{
			const Tload_group lightest = groups.top();
			groups.pop();
			owners[i] = lightest.second;
			groups.push(std::make_pair(lightest.first+item_weights[i], lightest.second));
		}
		return owners;
	}

	/*
	traversal_period
		In: [2,3]
		Out: [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]]
	*/
	template<typename Tcell>
	std::vector<std::array<Tcell,1>> traversal_period(const std::array<Tcell,1> &period)
	{
		std::vector<std::array<Tcell,1>> cells;
		cells.reserve(std::accumulate( period.begin(), period.end(), 1, std::multiplies<Tcell>() ));
		for(Tcell item=0; item<period[0]; ++item)
			cells.push_back(std::array<Tcell,1>{item});
		return cells;
	}

	template<typename Tcell, std::size_t Ndim>
	std::vector<std::array<Tcell,Ndim>> traversal_period(const std::array<Tcell,Ndim> &period)
	{
		auto cat_array = [](
			const Tcell &item_first,
			const std::array<Tcell,Ndim-1> &array_latter)
		-> std::array<Tcell,Ndim>
		{
			std::array<Tcell,Ndim> array_all;
			array_all[0] = item_first;
			for(std::size_t i=1; i<Ndim; ++i)
				array_all[i] = array_latter[i-1];
			return array_all;
		};

		auto sub_array = [](
			const std::array<Tcell,Ndim> &array_all)
		-> std::array<Tcell,Ndim-1>
		{
			std::array<Tcell,Ndim-1> array_sub;
			for(std::size_t i=1; i<Ndim; ++i)
				array_sub[i-1] = array_all[i];
			return array_sub;
		};

		std::vector<std::array<Tcell, Ndim>> cells;
		cells.reserve(std::accumulate( period.begin(), period.end(), 1, std::multiplies<Tcell>() ));
		const std::array<Tcell,Ndim-1> period_sub = sub_array(period);
		const std::vector<std::array<Tcell, Ndim-1>> cells_sub = traversal_period(period_sub);
		for(Tcell item_first=0; item_first<period[0]; ++item_first)
			for(const auto &cell_sub : cells_sub)
				cells.push_back(cat_array(item_first, cell_sub));
		return cells;
	}

	/*
	traversal_atom_period
		In: [A0,A1] [3]
		Out: [[A0,0], [A0,1], [A0,2], [A1,0], [A1,1], [A1,2]]
	*/
	template<typename TA, typename Tcell, std::size_t Ndim>
	std::vector<std::pair<TA,std::array<Tcell,Ndim>>> traversal_atom_period(
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period)
	{
		using TC = std::array<Tcell,Ndim>;
		using TAC = std::pair<TA,TC>;
		const std::vector<TC> cells_origin = traversal_period(period);
		const std::vector<TC> cells = Global_Func::mod_period(cells_origin, period);
		std::vector<TAC> atoms_periods;
		atoms_periods.reserve( atoms.size() * cells.size() );
		for(const TA &atom : atoms)
			for(const TC &cell : cells)
				atoms_periods.push_back(std::make_pair(atom,cell));
		return atoms_periods;
	}

	template<typename TA>
	std::vector<TA> divide_atoms(
		const std::size_t group_rank,
		const std::size_t group_size,
		const std::vector<TA> &atoms)
	{
		const std::size_t mod = atoms.size() % group_size;
		const std::size_t index_begin =
			(group_rank < mod)
			? group_rank * (atoms.size()/group_size+1)
			: mod * (atoms.size()/group_size+1) + (group_rank-mod) * (atoms.size()/group_size);
		const std::size_t index_size =
			(group_rank < mod)
			? atoms.size()/group_size+1
			: atoms.size()/group_size;
		return std::vector<TA>(atoms.begin()+index_begin, atoms.begin()+index_begin+index_size);
	}

	template<typename TA>
	std::vector<TA> divide_atoms(
		const std::size_t group_rank,
		const std::size_t group_size,
		const std::vector<TA> &atoms,
		const std::map<TA,std::size_t> &weights)
	{
		if(weights.empty())
			return divide_atoms(group_rank, group_size, atoms);

		std::vector<std::size_t> item_weights;
		item_weights.reserve(atoms.size());
		for(const TA &atom : atoms)
			item_weights.push_back(atom_weight(weights, atom));

		const std::vector<std::size_t> owners = assign_owners_weighted(group_size, item_weights);

		// get the atoms distributed onto the current rank
		std::vector<TA> atoms_divide;
		for(std::size_t i=0; i<atoms.size(); ++i)
			if(owners[i]==group_rank)
				atoms_divide.push_back(atoms[i]);
		return atoms_divide;
	}

	template<typename TA, typename Tcell, std::size_t Ndim>
	std::vector<std::pair<TA,std::array<Tcell,Ndim>>> divide_atoms(
		const std::size_t group_rank,
		const std::size_t group_size,
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period)
	{
		const std::vector<TA> atoms_divide = divide_atoms(group_rank, group_size, atoms);
		return traversal_atom_period(atoms_divide, period);
	}

	template<typename TA, typename Tcell, std::size_t Ndim>
	std::vector<std::pair<TA,std::array<Tcell,Ndim>>> divide_atoms(
		const std::size_t group_rank,
		const std::size_t group_size,
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period,
		const std::map<TA,std::size_t> &weights)
	{
		const std::vector<TA> atoms_divide = divide_atoms(group_rank, group_size, atoms, weights);
		return traversal_atom_period(atoms_divide, period);
	}

	template<typename TA, typename Tcell, std::size_t Ndim>
	std::vector<std::pair<TA,std::array<Tcell,Ndim>>> divide_atoms_periods(
		const std::size_t group_rank,
		const std::size_t group_size,
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period)
	{
		using TC = std::array<Tcell,Ndim>;
		using TAC = std::pair<TA,TC>;
		const std::vector<TC> cells_origin = traversal_period(period);
		const std::vector<TC> cells = Global_Func::mod_period(cells_origin, period);
		const std::size_t mod = atoms.size() * cells.size() % group_size;
		const std::size_t index_begin =
			(group_rank < mod)
			? group_rank * (atoms.size()*cells.size()/group_size+1)
			: mod * (atoms.size()*cells.size()/group_size+1) + (group_rank-mod) * (atoms.size()*cells.size()/group_size);
		const std::size_t index_size =
			(group_rank < mod)
			? atoms.size()*cells.size()/group_size+1
			: atoms.size()*cells.size()/group_size;
		const std::size_t index_end = index_begin + index_size;

		std::vector<TAC> atoms_periods_divide;
		atoms_periods_divide.reserve(index_size);

		const std::size_t iatom_begin = index_begin / cells.size();
		std::size_t index = iatom_begin * cells.size();
		for(std::size_t iatom=iatom_begin; iatom<atoms.size(); ++iatom)
		{
			for(const TC &cell : cells)
			{
				if(index>=index_begin)
					atoms_periods_divide.push_back(std::make_pair(atoms[iatom], cell));
				++index;
				if(index>=index_end)
					return atoms_periods_divide;
			}
		}
		throw std::range_error(std::string(__FILE__)+" line "+std::to_string(__LINE__));
	}

	template<typename TA, typename Tcell, std::size_t Ndim>
	std::vector<std::pair<TA,std::array<Tcell,Ndim>>> divide_atoms_periods(
		const std::size_t group_rank,
		const std::size_t group_size,
		const std::vector<TA> &atoms,
		const std::array<Tcell,Ndim> &period,
		const std::map<TA,std::size_t> &weights)
	{
		if(weights.empty())
			return divide_atoms_periods(group_rank, group_size, atoms, period);

		using TC = std::array<Tcell,Ndim>;
		using TAC = std::pair<TA,TC>;
		const std::vector<TC> cells_origin = traversal_period(period);
		const std::vector<TC> cells = Global_Func::mod_period(cells_origin, period);

		std::vector<TAC> atoms_periods;
		std::vector<std::size_t> item_weights;
		atoms_periods.reserve( atoms.size() * cells.size() );
		item_weights.reserve( atoms.size() * cells.size() );
		for(const TA &atom : atoms)
		{
			const std::size_t weight = atom_weight(weights, atom);
			for(const TC &cell : cells)
			{
				atoms_periods.push_back(std::make_pair(atom,cell));
				item_weights.push_back(weight);
			}
		}

		const std::vector<std::size_t> owners = assign_owners_weighted(group_size, item_weights);

		std::vector<TAC> atoms_periods_divide;
		for(std::size_t i=0; i<atoms_periods.size(); ++i)
			if(owners[i]==group_rank)
				atoms_periods_divide.push_back(atoms_periods[i]);
		return atoms_periods_divide;
	}
}

}