#pragma once

#include "RI/global/Map_Operator-2.h"
#include "RI/global/Map_Operator-3.h"
#include "unittests/print_stl.h"

#include <string>

namespace Map_Operator_Test
{
	void test_union_map1()
	{
		std::map<int,double> m1, m2;
		m1[2]=2;	m1[1]=1;
		m2[0]=0;	m2[1]=10;
		const std::function<double(const double&,const double&)> plus = std::plus<double>();
		std::cout<<Map_Operator::zip_union(m1,m2,plus)<<std::endl;
		/*{
			0: 0,
			1: 11,
			2: 2
		}*/
	}

	void test_union_map2()
	{
		std::map<int,std::map<std::string,double>> m1, m2;
		m1[2]["a"]=2;	m1[1]["b"]=1;
		m2[0]["c"]=0;	m2[1]["b"]=10;	m1[1]["c"]=20;	
		const std::function<double(const double&,const double&)> plus = std::plus<double>();
		std::cout<<Map_Operator::zip_union(m1,m2,plus)<<std::endl;
		/*{
			0: {"c":0},
			1: {"b":11, "c":20},
			2: {"a":2}
		}*/
	}	

	void test_intersection_map1()
	{
		std::map<int,double> m1, m2;
		m1[2]=2;	m1[1]=1;
		m2[0]=0;	m2[1]=10;
		const std::function<double(const double&,const double&)> plus = std::plus<double>();
		std::cout<<Map_Operator::zip_intersection(m1,m2,plus)<<std::endl;
		/*{
			1: 11
		}*/
	}

	void test_intersection_map2()
	{
		std::map<int,std::map<std::string,double>> m1, m2;
		m1[2]["a"]=2;	m1[1]["b"]=1;
		m2[0]["c"]=0;	m2[1]["b"]=10;	m1[1]["c"]=20;	
		const std::function<double(const double&,const double&)> plus = std::plus<double>();
		std::cout<<Map_Operator::zip_intersection(m1,m2,plus)<<std::endl;
		/*{
			1: {"b":11},
		}*/
	}
	
	void test_transform_map()
	{
		std::map<int,std::map<std::string,double>> m;
		m[0]["c"]=0;	m[1]["b"]=10;	m[1]["c"]=20;
		constexpr double frac = 2;
		const std::function<double(const double&)> multiply_frac = [&frac](const double &t){ return t*frac; };
		std::cout<<Map_Operator::transform(m, multiply_frac)<<std::endl;
		/*{
			0: {"c":0},
			1: {"b":20, "c":40}
		}*/		
	}

	void test_for_each_map()
	{
		std::map<int,std::map<std::string,double>> m;
		m[0]["c"]=0;	m[1]["b"]=10;	m[1]["c"]=20;
		constexpr double frac = 2;
		const std::function<void(double&)> multiply_frac = [&frac](double &t){ t*=frac; };
		Map_Operator::for_each(m, multiply_frac);
		std::cout<<m<<std::endl;
		/*{
			0: {"c":0},
			1: {"b":20, "c":40}
		}*/		
	}
}