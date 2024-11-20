#pragma once
/*template <typename IteratorB,typename IteratorE>
void advance_all (IteratorB & begin,IteratorE & end) {
	++begin;
}
template <typename IteratorB,typename IteratorE, typename ... Iterators>
void advance_all (IteratorB & begin,IteratorE & end, Iterators& ... iterators){
	++begin;
	advance_all(iterators...);
}
template <typename IteratorB,typename IteratorE>
bool compare_all (IteratorB & begin,IteratorE & end) {
	begin != end;
}
template <typename IteratorB,typename IteratorE, typename ... Iterators>
bool compare_all (IteratorB & begin,IteratorE & end, Iterators& ... iterators){
	return begin != end && compare_all(iterators...);
}
template <typename Function, typename IteratorB,typename IteratorE, typename ... Iterators>
Function zip_for_each (Function func, IteratorB begin, IteratorE end, Iterators ... iterators){
	for(;begin != end && compare_all(iterators...); ++begin, advance_all(iterators...))
		func(*begin, *(iterators)... );

	return func;
}
template<typename Function,typename ContTy1,typename ContTy2>
Function zip_range(Function func, ContTy1 Cont1,ContTy2 Cont2){
	return zip_for_each(func,Cont1.begin(),Cont1.end(),Cont2.begin(),Cont2.end());
}*/
template <typename Function, typename ItTy1,typename ItTy2>
void for_each (ItTy1 begin1, ItTy1 end1, ItTy2 begin2, ItTy2 end2,Function func){
	for (;begin1 != end1 && begin2 != end2; ++begin1,++begin2){
                func(*begin1,*begin2);
	}
}
template <typename Function, typename ItTy1,typename ItTy2,typename ItTy3>
void for_each (ItTy1 begin1, ItTy1 end1, ItTy2 begin2, ItTy2 end2, ItTy3 begin3, ItTy3 end3,Function func){
        for (;begin1 != end1 && begin2 != end2 && begin3 != end3; ++begin1,++begin2,++begin3){
                func(*begin1,*begin2,*begin3);
	}
}
template<typename Function,typename ContTy1,typename ContTy2>
Function zip_range(ContTy1 Cont1,ContTy2 Cont2,Function func){
    return for_each(Cont1.begin(),Cont1.end(),Cont2.begin(),Cont2.end(),func);
}
