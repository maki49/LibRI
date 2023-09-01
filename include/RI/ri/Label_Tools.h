// ===================
//  Author: Peize Lin
//  date: 2022.10.21
// ===================

#pragma once

#include "Label.h"
#include <string>
#include <vector>
#include <stdexcept>

namespace RI
{

namespace Label_Tools
{
	static std::string get_name(const Label::ab &label)
	{
		switch(label)
		{
			case Label::ab::a:		return "a";
			case Label::ab::b:		return "b";
			case Label::ab::a0b0:	return "a0b0";
			case Label::ab::a0b1:	return "a0b1";
			case Label::ab::a0b2:	return "a0b2";
			case Label::ab::a1b0:	return "a1b0";
			case Label::ab::a1b1:	return "a1b1";
			case Label::ab::a1b2:	return "a1b2";
			case Label::ab::a2b0:	return "a2b0";
			case Label::ab::a2b1:	return "a2b1";
			case Label::ab::a2b2:	return "a2b2";
			default:	throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
		}
	}

	static std::string get_name(const Label::ab_ab &label)
	{
		switch(label)
		{
			case Label::ab_ab::a1b1_a2b2:		return "a1b1_a2b2";
			case Label::ab_ab::a1b0_a2b2:		return "a1b0_a2b2";
			case Label::ab_ab::a1b0_a2b1:		return "a1b0_a2b1";
			case Label::ab_ab::a0b1_a2b2:		return "a0b1_a2b2";
			case Label::ab_ab::a0b0_a2b2:		return "a0b0_a2b2";
			case Label::ab_ab::a0b0_a2b1:		return "a0b0_a2b1";
			case Label::ab_ab::a0b1_a1b2:		return "a0b1_a1b2";
			case Label::ab_ab::a0b0_a1b2:		return "a0b0_a1b2";
			case Label::ab_ab::a0b0_a1b1:		return "a0b0_a1b1";
			case Label::ab_ab::a1b2_a2b1:		return "a1b2_a2b1";
			case Label::ab_ab::a1b2_a2b0:		return "a1b2_a2b0";
			case Label::ab_ab::a1b1_a2b0:		return "a1b1_a2b0";
			case Label::ab_ab::a0b2_a2b1:		return "a0b2_a2b1";
			case Label::ab_ab::a0b2_a2b0:		return "a0b2_a2b0";
			case Label::ab_ab::a0b1_a2b0:		return "a0b1_a2b0";
			case Label::ab_ab::a0b2_a1b1:		return "a0b2_a1b1";
			case Label::ab_ab::a0b2_a1b0:		return "a0b2_a1b0";
			case Label::ab_ab::a0b1_a1b0:		return "a0b1_a1b0";
			default:	throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
		}
	}

	template<typename Tlabel>
	std::string get_name(const std::vector<Tlabel> &label_list)
	{
		std::string name = "";
		for(const auto &label : label_list)
			name += Label_Tools::get_name(label) + "_";
		return name.substr(0, name.size()-1);
	}
}

}