#pragma once

#include "../../global/Array_Operator.h"

#include <array>
#include <map>
#include <set>

#define NO_SEC_RETURN_TRUE if(this->irreducible_sector_.empty()) return true;
#define NO_QUADS_RETURN_TRUE if(this->irreducible_quads_.empty()) return true;

namespace RI
{
	template<typename TA, typename TC, typename Tdata>
	class Symmetry_Filter
	{
		using TAC = std::pair<TA, TC>;
		using Tab = std::pair<TA, TA>;
		using TabR = std::pair<Tab, TC>;
		using Tsec = std::map<Tab, std::set<TC>>;
		// for irreducible quads
		using Tquad_abR = std::pair<TabR, TC>;
		using Tquads = std::map<TabR, std::set<Tquad_abR>>;
		using Tquads_weight = std::map<TabR, std::map<Tquad_abR, int>>;

	  public:
		  Symmetry_Filter(const TC& period_in, const Tsec& irsec,
			  const Tquads& irquads = {}, const Tquads_weight& irquads_weight = {})
			  :period(period_in), irreducible_sector_(irsec),
			  irreducible_quads_(irquads), irreducible_quads_weight_(irquads_weight) {
		  }
		bool in_irreducible_sector(const TA& Aa, const TAC& Ab) const
		{
			NO_SEC_RETURN_TRUE;
			using namespace Array_Operator;
			const Tab& ap = { Aa, Ab.first };
			if (irreducible_sector_.find(ap) != irreducible_sector_.end())
				if (irreducible_sector_.at(ap).find(Ab.second % this->period) != irreducible_sector_.at(ap).end())
					return true;
			return false;
		}
		bool in_irreducible_sector(const TAC& Aa, const TAC& Ab) const
		{
			NO_SEC_RETURN_TRUE;
			using namespace Array_Operator;
			const TC dR = (Ab.second - Aa.second) % this->period;
			const std::pair<TA, TA> ap = { Aa.first, Ab.first };
			if (irreducible_sector_.find(ap) != irreducible_sector_.end())
				if (irreducible_sector_.at(ap).find(dR) != irreducible_sector_.at(ap).end())
					return true;
			return false;
		}
		bool is_Aa_in_irreducible_sector(const TA& Aa) const
		{
			NO_SEC_RETURN_TRUE;
			for (const auto& apRs : irreducible_sector_)
				if (apRs.first.first == Aa)return true;
			return false;
		}
		bool is_Ab_in_irreducible_sector(const TA& Ab) const
		{
			NO_SEC_RETURN_TRUE;
			for (const auto& apRs : irreducible_sector_)
				if (apRs.first.second == Ab)return true;
			return false;
		}
		TabR get_abR(const TA& Aa, const TAC& Ab) const
		{
			using namespace Array_Operator;
			return { {Aa,Ab.first}, Ab.second % this->period };
		}
		TabR get_abR(const TAC& Aa, const TAC& Ab) const
		{
			using namespace Array_Operator;
			return { {Aa.first,Ab.first}, (Ab.second - Aa.second) % this->period };
		}

		// for irreducible quads
		bool is_Aa01_Aa2_Ab2_in_irreducible_quads(const TA& Aa01, const TAC& Aa2, const TAC& Ab2) const
		{
			using namespace Array_Operator;
			return is_Aa01_Aa2_Ab2_in_irreducible_quads(Aa01, get_abR(Aa2, Ab2), Aa2.second % this->period);
		}
		bool is_Aa01_Aa2_Ab2_in_irreducible_quads(const TAC& Aa01, const TA& Aa2, const TAC& Ab2) const
		{
			using namespace Array_Operator;
			return is_Aa01_Aa2_Ab2_in_irreducible_quads(Aa01.first, get_abR(Aa2, Ab2), (-Aa01.second) % this->period);
		}
		bool is_Ab01_Aa2_Ab2_in_irreducible_quads(const TA& Ab01, const TAC& Aa2, const TAC& Ab2) const
		{
			using namespace Array_Operator;
			return is_Ab01_Aa2_Ab2_in_irreducible_quads(Ab01, get_abR(Aa2, Ab2), Aa2.second % this->period);
		}
		bool is_Ab01_Aa2_Ab2_in_irreducible_quads(const TAC& Ab01, const TA& Aa2, const TAC& Ab2) const
		{
			using namespace Array_Operator;
			return is_Ab01_Aa2_Ab2_in_irreducible_quads(Ab01.first, get_abR(Aa2, Ab2), (-Ab01.second) % this->period);
		}
		bool is_irreducible_quad(const TabR& Aa01_Ab01, const TabR& Aa2_Ab2, const TC& R_a01_a2) const
		{
			NO_SEC_RETURN_TRUE;
			NO_QUADS_RETURN_TRUE;
			auto find_ab01 = irreducible_quads_.find(Aa01_Ab01);
			if (find_ab01 == irreducible_quads_.end())
				return false;
			return find_ab01->second.find({ Aa2_Ab2, R_a01_a2 }) != find_ab01->second.end();
		}
		bool is_Aa01_Aa2_Ab2_in_irreducible_quads(const TA& Aa01, const TabR& Aa2_Ab2, const TC& R_a01_a2) const
		{
			NO_SEC_RETURN_TRUE;
			NO_QUADS_RETURN_TRUE;
			for (auto& apRs : irreducible_sector_)
				if (apRs.first.first == Aa01)
					for (auto& R_a01_b01 : apRs.second)
						if (is_irreducible_quad({ apRs.first, R_a01_b01 }, Aa2_Ab2, R_a01_a2))
							return true;
			return false;
		}
		bool is_Ab01_Aa2_Ab2_in_irreducible_quads(const TA& Ab01, const TabR& Aa2_Ab2, const TC& R_b01_a2) const
		{
			NO_SEC_RETURN_TRUE;
			NO_QUADS_RETURN_TRUE;
			using namespace Array_Operator;
			for (auto& apRs : irreducible_sector_)
				if (apRs.first.second == Ab01)
					for (auto& R_a01_b01 : apRs.second)
						if (is_irreducible_quad({ apRs.first, R_a01_b01 }, Aa2_Ab2, (R_a01_b01 + R_b01_a2) % period))
							return true;
			return false;
		}

	  public:	// private:
		const TC& period;
		const Tsec& irreducible_sector_;
		const Tquads irreducible_quads_;
		const Tquads_weight irreducible_quads_weight_;
	};

}

#undef NO_SEC_RETURN_TRUE