#pragma once
template <typename ItTy1,typename ItTy2,typename Function>
void for_each (ItTy1 begin1, ItTy1 end1, ItTy2 begin2, ItTy2 end2,Function func){
	for (;begin1 != end1 && begin2 != end2; ++begin1,++begin2){
		func(*begin1, *begin2);
	}
}
template <typename ItTy1,typename ItTy2,typename ItTy3,typename Function>
void for_each (ItTy1 begin1, ItTy1 end1, ItTy2 begin2, ItTy2 end2, ItTy3 begin3, ItTy3 end3,Function func){
	for (;begin1 != end1 && begin2 != end2 && begin3 != end3; ++begin1,++begin2,++begin3){
		func(*begin1, *begin2, *begin3);
	}
}
template <typename ItTy1,typename ItTy2,typename ItTy3,typename ItTy4,typename Function>
void for_each (ItTy1 begin1, ItTy1 end1, ItTy2 begin2, ItTy2 end2, ItTy3 begin3, ItTy3 end3, ItTy4 begin4, ItTy4 end4,Function func){
	for (;begin1 != end1 && begin2 != end2 && begin3 != end3 && begin4 != end4; ++begin1,++begin2,++begin3,++begin4){
		func(*begin1, *begin2, *begin3,*begin4);
	}
}
template <typename ContTy1,typename ContTy2,typename Function>
void for_each_range (ContTy1 & Cont1, ContTy2 & Cont2,Function func){
	for_each(Cont1.begin(),Cont1.end(),Cont2.begin(),Cont2.end(),func);
}
template <typename ContTy1,typename ContTy2,typename ContTy3,typename Function>
void for_each_range (ContTy1 & Cont1, ContTy2 & Cont2,ContTy3 & Cont3,Function func){
	for_each(Cont1.begin(),Cont1.end(),Cont2.begin(),Cont2.end(),Cont3.begin(),Cont3.end(),func);
}
template <typename ContTy1,typename ContTy2,typename ContTy3,typename ContTy4,typename Function>
void for_each_range (ContTy1 & Cont1, ContTy2 & Cont2,ContTy3 & Cont3,ContTy4 & Cont4,Function func){
	for_each(Cont1.begin(),Cont1.end(),Cont2.begin(),Cont2.end(),Cont3.begin(),Cont3.end(),Cont4.begin(),Cont4.end(),func);
}
