#pragma once
#include <utility>
namespace std{
	template <typename f_ty,typename s_ty>
	struct _unpair_helper{
		f_ty * first;
		s_ty * second;
		void operator = (std::pair<f_ty,s_ty> & in_val){
			*this->first = in_val.first;
			*this->second = in_val.second;
		}
	};
	template <typename f_ty,typename s_ty>
	_unpair_helper<f_ty,s_ty> unpair(f_ty & first,s_ty & second){
		return _unpair_helper<f_ty,s_ty>{&first,&second};
	}
};
