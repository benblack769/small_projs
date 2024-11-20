#pragma once
#include <vector>
template<typename ret_ty,typename input_ty, typename fn_ty>
ret_ty construct_vec(const input_ty & input,fn_ty conv_fn){
	ret_ty res(input.size());
	for(std::size_t i = 0; i < input.size(); i++){
		res[i] = conv_fn(input[i]);
	}
	return res;
}
