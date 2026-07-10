// ===================
//  Author: Peize Lin
//  date: 2022.07.13
// ===================

#pragma once

#include "RI/distribute/Divide_Atoms.h"
#include "unittests/print_stl.h"

namespace Divide_Atoms_Test
{
	static void test_divide_atoms()
	{
		const std::size_t group_size = 6;
		std::vector<std::size_t> atoms(31);
		for(std::size_t i=0; i<atoms.size(); ++i)
			atoms[i]=i;
		for(std::size_t i=0; i<group_size; ++i)
			std::cout<<RI::Divide_Atoms::divide_atoms(i, group_size, atoms)<<std::endl;
	}
	/*
		0|	1|	2|	3|	4|	5|
		6|	7|	8|	9|	10|
		11|	12|	13|	14|	15|
		16|	17|	18|	19|	20|
		21|	22|	23|	24|	25|
		26|	27|	28|	29|	30|
	*/


	static void test_divide_atoms_with_period()
	{
		const std::size_t group_size = 6;
		std::vector<std::size_t> atoms(31);
		for(std::size_t i=0; i<atoms.size(); ++i)
			atoms[i]=i;
		const std::array<std::size_t,1> period = {2};
		for(std::size_t i=0; i<group_size; ++i)
			std::cout<<RI::Divide_Atoms::divide_atoms(i, group_size, atoms, period)<<std::endl;
	}
	/*
		{ 0, 0	 }|	{ 0, 1	 }|	{ 1, 0	 }|	{ 1, 1	 }|	{ 2, 0	 }|	{ 2, 1	 }|	{ 3, 0	 }|	{ 3, 1	 }|	{ 4, 0	 }|	{ 4, 1	 }|	{ 5, 0	 }|	{ 5, 1	 }|
		{ 6, 0	 }|	{ 6, 1	 }|	{ 7, 0	 }|	{ 7, 1	 }|	{ 8, 0	 }|	{ 8, 1	 }|	{ 9, 0	 }|	{ 9, 1	 }|	{ 10, 0	 }|	{ 10, 1	 }|
		{ 11, 0	 }|	{ 11, 1	 }|	{ 12, 0	 }|	{ 12, 1	 }|	{ 13, 0	 }|	{ 13, 1	 }|	{ 14, 0	 }|	{ 14, 1	 }|	{ 15, 0	 }|	{ 15, 1	 }|
		{ 16, 0	 }|	{ 16, 1	 }|	{ 17, 0	 }|	{ 17, 1	 }|	{ 18, 0	 }|	{ 18, 1	 }|	{ 19, 0	 }|	{ 19, 1	 }|	{ 20, 0	 }|	{ 20, 1	 }|
		{ 21, 0	 }|	{ 21, 1	 }|	{ 22, 0	 }|	{ 22, 1	 }|	{ 23, 0	 }|	{ 23, 1	 }|	{ 24, 0	 }|	{ 24, 1	 }|	{ 25, 0	 }|	{ 25, 1	 }|
		{ 26, 0	 }|	{ 26, 1	 }|	{ 27, 0	 }|	{ 27, 1	 }|	{ 28, 0	 }|	{ 28, 1	 }|	{ 29, 0	 }|	{ 29, 1	 }|	{ 30, 0	 }|	{ 30, 1	 }|
	*/


	static void test_divide_atoms_periods()
	{
		const std::size_t group_size = 6;
		std::vector<std::size_t> atoms(31);
		for(std::size_t i=0; i<atoms.size(); ++i)
			atoms[i]=i;
		const std::array<std::size_t,1> period = {2};
		for(std::size_t i=0; i<group_size; ++i)
			std::cout<<RI::Divide_Atoms::divide_atoms_periods(i, group_size, atoms, period)<<std::endl;
	}
	/*
		{ 0, 0	 }|	{ 0, 1	 }|	{ 1, 0	 }|	{ 1, 1	 }|	{ 2, 0	 }|	{ 2, 1	 }|	{ 3, 0	 }|	{ 3, 1	 }|	{ 4, 0	 }|	{ 4, 1	 }|	{ 5, 0	 }|
		{ 5, 1	 }|	{ 6, 0	 }|	{ 6, 1	 }|	{ 7, 0	 }|	{ 7, 1	 }|	{ 8, 0	 }|	{ 8, 1	 }|	{ 9, 0	 }|	{ 9, 1	 }|	{ 10, 0	 }|	{ 10, 1	 }|
		{ 11, 0	 }|	{ 11, 1	 }|	{ 12, 0	 }|	{ 12, 1	 }|	{ 13, 0	 }|	{ 13, 1	 }|	{ 14, 0	 }|	{ 14, 1	 }|	{ 15, 0	 }|	{ 15, 1	 }|
		{ 16, 0	 }|	{ 16, 1	 }|	{ 17, 0	 }|	{ 17, 1	 }|	{ 18, 0	 }|	{ 18, 1	 }|	{ 19, 0	 }|	{ 19, 1	 }|	{ 20, 0	 }|	{ 20, 1	 }|
		{ 21, 0	 }|	{ 21, 1	 }|	{ 22, 0	 }|	{ 22, 1	 }|	{ 23, 0	 }|	{ 23, 1	 }|	{ 24, 0	 }|	{ 24, 1	 }|	{ 25, 0	 }|	{ 25, 1	 }|
		{ 26, 0	 }|	{ 26, 1	 }|	{ 27, 0	 }|	{ 27, 1	 }|	{ 28, 0	 }|	{ 28, 1	 }|	{ 29, 0	 }|	{ 29, 1	 }|	{ 30, 0	 }|	{ 30, 1	 }|
	*/


	// 27 light atoms (nao=2) followed by 4 heavy ones (nao=25): sum(nao) = 154.
	// Balancing the atom count would give the last group 5 heavy-ish atoms, i.e. 4x the load
	// of the first. Balancing sum(nao) gives every group 25 or 26, at the price of
	// non-contiguous groups.
	static std::map<std::size_t,std::size_t> nao_heavy_tail()
	{
		std::map<std::size_t,std::size_t> nao;
		for(std::size_t i=0; i<27; ++i)	nao[i] = 2;
		for(std::size_t i=27; i<31; ++i)	nao[i] = 25;
		return nao;
	}

	static void test_divide_atoms_nao()
	{
		const std::size_t group_size = 6;
		std::vector<std::size_t> atoms(31);
		for(std::size_t i=0; i<atoms.size(); ++i)
			atoms[i]=i;
		const std::map<std::size_t,std::size_t> nao = nao_heavy_tail();
		for(std::size_t i=0; i<group_size; ++i)
			std::cout<<RI::Divide_Atoms::divide_atoms(i, group_size, atoms, nao)<<std::endl;
	}
	/*
		26|	27|
		28|
		29|
		30|
		0|	2|	4|	6|	8|	10|	12|	14|	16|	18|	20|	22|	24|
		1|	3|	5|	7|	9|	11|	13|	15|	17|	19|	21|	23|	25|

		sum(nao):    27|	25|	25|	25|	26|	26|	  (total 154, ideal 25.67)
		atom count:   2|	 1|	 1|	 1|	13|	13|
		The atom counts are deliberately lopsided: each heavy atom alone is worth
		12 light ones, so equal load means unequal counts.
	*/


	static void test_divide_atoms_periods_nao()
	{
		const std::size_t group_size = 6;
		std::vector<std::size_t> atoms(31);
		for(std::size_t i=0; i<atoms.size(); ++i)
			atoms[i]=i;
		const std::array<int,1> period = {2};
		const std::map<std::size_t,std::size_t> nao = nao_heavy_tail();
		for(std::size_t i=0; i<group_size; ++i)
			std::cout<<RI::Divide_Atoms::divide_atoms_periods(i, group_size, atoms, period, nao)<<std::endl;
	}
	/*
		Every {atom,cell} is one item of weight nao[atom]; sum over all 62 items = 308.
		Unlike divide_atoms above, an atom's two cells may land in different groups.

		{ 26, 0	 }|	{ 27, 0	 }|	{ 30, 0	 }|
		{ 26, -1	 }|	{ 27, -1	 }|	{ 30, -1	 }|
		{ 0, 0	 }|	{ 2, 0	 }|	{ 4, 0	 }|	{ 6, 0	 }|	{ 8, 0	 }|	{ 10, 0	 }|	{ 12, 0	 }|	{ 14, 0	 }|	{ 16, 0	 }|	{ 18, 0	 }|	{ 20, 0	 }|	{ 22, 0	 }|	{ 24, 0	 }|	{ 28, 0	 }|
		{ 0, -1	 }|	{ 2, -1	 }|	{ 4, -1	 }|	{ 6, -1	 }|	{ 8, -1	 }|	{ 10, -1	 }|	{ 12, -1	 }|	{ 14, -1	 }|	{ 16, -1	 }|	{ 18, -1	 }|	{ 20, -1	 }|	{ 22, -1	 }|	{ 24, -1	 }|	{ 28, -1	 }|
		{ 1, 0	 }|	{ 3, 0	 }|	{ 5, 0	 }|	{ 7, 0	 }|	{ 9, 0	 }|	{ 11, 0	 }|	{ 13, 0	 }|	{ 15, 0	 }|	{ 17, 0	 }|	{ 19, 0	 }|	{ 21, 0	 }|	{ 23, 0	 }|	{ 25, 0	 }|	{ 29, 0	 }|
		{ 1, -1	 }|	{ 3, -1	 }|	{ 5, -1	 }|	{ 7, -1	 }|	{ 9, -1	 }|	{ 11, -1	 }|	{ 13, -1	 }|	{ 15, -1	 }|	{ 17, -1	 }|	{ 19, -1	 }|	{ 21, -1	 }|	{ 23, -1	 }|	{ 25, -1	 }|	{ 29, -1	 }|

		sum(nao):	52|	52|	51|	51|	51|	51|	  (total 308, ideal 51.33)
	*/
}