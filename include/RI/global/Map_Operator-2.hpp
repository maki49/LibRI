// ====================
//  Author: Peize Lin
//  Date:   2022.07.27
// ====================

#pragma once

#include "Map_Operator-2.h"

namespace Map_Operator
{
	template<typename Tkey, typename Tvalue>
	std::map<Tkey,Tvalue> zip_union_prototype(
		const std::map<Tkey,Tvalue> &m1,
		const std::map<Tkey,Tvalue> &m2,
		const std::function<Tvalue(const Tvalue&,const Tvalue&)> &func_prototype)
	{
		std::map<Tkey,Tvalue> m;
		auto ptr1 = m1.begin();
		auto ptr2 = m2.begin();
		while(ptr1!=m1.end() && ptr2!=m2.end())
		{
			if(ptr1->first == ptr2->first)
			{
				m.emplace_hint(m.end(), ptr1->first, func_prototype(ptr1->second, ptr2->second));
				++ptr1;
				++ptr2;
			}
			else if(ptr1->first < ptr2->first)
			{
				m.emplace_hint(m.end(), ptr1->first, ptr1->second);
				++ptr1;
			}
			else
			{
				m.emplace_hint(m.end(), ptr2->first, ptr2->second);
				++ptr2;
			}
		}
		m.insert(ptr1, m1.end());
		m.insert(ptr2, m2.end());
		return m;
	}

	template<typename Tkey, typename Tdata>
	std::map<Tkey,Tdata> zip_union(
		const std::map<Tkey,Tdata> &m1,
		const std::map<Tkey,Tdata> &m2,
		const std::function<Tdata(const Tdata&,const Tdata&)> &func)
	{
		const std::function<Tdata(const Tdata&, const Tdata&)> 
			func_prototype = [&func](const Tdata& v1, const Tdata &v2) -> Tdata
			{ return func(v1, v2); };
		return zip_union_prototype(m1, m2, func_prototype);
	}

	template<typename Tkey1, typename Tkey2, typename Tvalue, typename Tdata>
	std::map<Tkey1,std::map<Tkey2,Tvalue>> zip_union(
		const std::map<Tkey1,std::map<Tkey2,Tvalue>> &m1,
		const std::map<Tkey1,std::map<Tkey2,Tvalue>> &m2,
		const std::function<Tdata(const Tdata&,const Tdata&)> &func)
	{
		using Tvalue1 = std::map<Tkey2,Tvalue>;
		const std::function<Tvalue1(const Tvalue1&, const Tvalue1&)> 
			func_prototype = [&func](const Tvalue1& v1, const Tvalue1 &v2) -> Tvalue1
			{ return zip_union(v1, v2, func); };
		return zip_union_prototype(m1, m2, func_prototype);
	}

	template<typename Tkey, typename Tvalue, typename Tdata>
	std::map<Tkey,Tvalue> zip_intersection(
		const std::map<Tkey,Tvalue> &m1,
		const std::map<Tkey,Tvalue> &m2,
		const std::function<Tdata(const Tdata&,const Tdata&)> &func);

	template<typename Tkey, typename Tvalue>
	std::map<Tkey,Tvalue> zip_intersection_prototype(
		const std::map<Tkey,Tvalue> &m1,
		const std::map<Tkey,Tvalue> &m2,
		const std::function<Tvalue(const Tvalue&,const Tvalue&)> &func_prototype)
	{
		std::map<Tkey,Tvalue> m;
		if(m1.size()<m2.size())
		{
			for(const auto &i1 : m1)
			{
				const auto ptr2 = m2.find(i1.first);
				if(ptr2!=m2.end())
					m.emplace_hint(m.end(), i1.first, func_prototype(i1.second, ptr2->second));
			}
		}
		else
		{
			for(const auto &i2 : m2)
			{
				const auto ptr1 = m1.find(i2.first);
				if(ptr1!=m1.end())
					m.emplace_hint(m.end(), i2.first, func_prototype(ptr1->second, i2.second));
			}
		}
		return m;
	}

	template<typename Tkey, typename Tdata>
	std::map<Tkey,Tdata> zip_intersection(
		const std::map<Tkey,Tdata> &m1,
		const std::map<Tkey,Tdata> &m2,
		const std::function<Tdata(const Tdata&,const Tdata&)> &func)
	{
		const std::function<Tdata(const Tdata&, const Tdata&)> 
			func_prototype = [&func](const Tdata& v1, const Tdata &v2)->Tdata
			{ return func(v1, v2); };
		return zip_intersection_prototype(m1, m2, func_prototype);
	}

	template<typename Tkey1, typename Tkey2, typename Tvalue, typename Tdata>
	std::map<Tkey1,std::map<Tkey2,Tvalue>> zip_intersection(
		const std::map<Tkey1,std::map<Tkey2,Tvalue>> &m1,
		const std::map<Tkey1,std::map<Tkey2,Tvalue>> &m2,
		const std::function<Tdata(const Tdata&,const Tdata&)> &func)
	{
		using Tvalue1 = std::map<Tkey2,Tvalue>;
		const std::function<Tvalue1(const Tvalue1&, const Tvalue1&)> 
			func_prototype = [&func](const Tvalue1& v1, const Tvalue1 &v2)
			{ return zip_intersection(v1, v2, func); };
		return zip_intersection_prototype(m1, m2, func_prototype);
	}		
} // namespace Map_Operator
