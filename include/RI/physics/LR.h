#pragma once
#include "./Exx.h"

namespace RI
{
// Nothing different from Exx, 
// Except the two density matrices can be different
template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
class  LR : public Exx
{
public:
	LR::LR(const std::string method = "cvc") { this->method = method; }	// Exx default mehtod: "loop3"

	void cal_force(const std::map<TA, std::map<TAC, Tensor<Tdata>>>& Ds_left,
		const std::array<std::string, 5>& save_names_suffix = { "","","","","" })	// "Cs","Vs","Ds","dCs","dVs"
	{
		// The only difference from Exx::cal_force: save Ds_left if not empty
		if (!Ds_left.empty())
			this->post_2D.saves["Ds_" + save_name_suffix] = this->post_2D.set_tensors_map2(Ds_left);
		this->cal_force(save_names_suffix);
	}
};

}