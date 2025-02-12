// ===================
//  Author: LUNASEA
//  date: 2024.06.08
// ===================

#pragma once

#include "../global/Global_Func-2.h"
#include "../global/Tensor.h"
#include "../ri/LRI.h"

#include <mpi.h>
#include <array>
#include <map>

namespace RI
{
	// 1. maybe we can construct a base class later 
	// and put the almost same functions like `set_Cs` in it

	// 2. usually the Hartree term does not need to be calculated by RI technique, 
	// but an implementation is still useful for benchmark

	template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
	class Hartree
	{
	public:
		using TC = std::array<Tcell, Ndim>;
		using TAC = std::pair<TA, TC>;
		using Tdata_real = Global_Func::To_Real_t<Tdata>;
		using Tpos = double;							// tmp
		constexpr static std::size_t Npos = Ndim;		// tmp
		using Tatom_pos = std::array<Tpos, Npos>;		// tmp

		void set_parallel(
			const MPI_Comm& mpi_comm,
			const std::map<TA, Tatom_pos>& atoms_pos,
			const std::array<Tatom_pos, Ndim>& latvec,
			const std::array<Tcell, Ndim>& period);

		void set_Cs(
			const std::map<TA, std::map<TAC, Tensor<Tdata>>>& Cs,
			const Tdata_real& threshold_C,
			const std::string& save_name_suffix = "");
		void set_Vs(
			const std::map<TA, std::map<TAC, Tensor<Tdata>>>& Vs,
			const Tdata_real& threshold_V,
			const std::string& save_name_suffix = "");
		void set_Ds(
			const std::map<TA, std::map<TAC, Tensor<Tdata>>>& Ds,
			const Tdata_real& threshold_D,
			const std::string& save_name_suffix = "");
		void set_csm_threshold(
			const Tdata_real& threshold) {
			this->lri.csm.set_threshold(threshold);
		}

		void cal_Hs(
			const std::array<std::string, 3>& save_names_suffix = { "","","" });		// "Cs","Vs","Ds"

		std::map<TA, std::map<TAC, Tensor<Tdata>>> Hs;
		Tdata energy = 0;
		std::array<std::map<TA, Tdata>, Ndim> force;
		Tensor<Tdata> stress;

		Exx_Post_2D<TA, TC, Tdata> post_2D;

	public:
		LRI<TA, Tcell, Ndim, Tdata> lri;

		struct Flag_Finish
		{
			bool stru = false;
			bool C = false;
			bool V = false;
			bool D = false;
		};
		Flag_Finish flag_finish;

		MPI_Comm mpi_comm;
		std::map<TA, Tatom_pos> atoms_pos;
		std::array<Tatom_pos, Ndim> latvec;
		std::array<Tcell, Ndim> period;
	};

}

#include "Hartree.hpp"