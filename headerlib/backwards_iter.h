#pragma once
template<typename cont_ty>
class reverse_obj{
public:
	cont_ty * cont_ptr;
	reverse_obj() = delete;
	reverse_obj(cont_ty & cont){
		cont_ptr = &cont;
	}
	auto begin()->decltype(cont_ptr->rbegin()){
		return cont_ptr->rbegin();
	}
	auto end()->decltype(cont_ptr->rend()){
		return cont_ptr->rend();
	}
};
template<typename cont_ty>
reverse_obj<cont_ty> reverse(cont_ty & cont){
	return reverse_obj<cont_ty>(cont);
}
//we need a seperate version for constant objects
template<typename cont_ty>
class reverse_obj_const{
public:
	cont_ty cont_ptr;
	reverse_obj_const() = delete;
	reverse_obj_const(reverse_obj_const & obj) = delete;
	reverse_obj_const(cont_ty && cont){
		cont_ptr = cont;
	}
	auto begin()->decltype(cont_ptr->rbegin()){
		return cont_ptr.rbegin();
	}
	auto end()->decltype(cont_ptr->rend()){
		return cont_ptr.rend();
	}
};
template<typename cont_ty>
reverse_obj_const<cont_ty> reverse(cont_ty && cont){
	return reverse_obj_const<cont_ty>(cont);
}
