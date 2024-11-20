#pragma once

#include <vector>
#include "point.h"

template<typename ArrayType>
class FArray2d
{
public:
	using ArrTy = std::vector<ArrayType>;
	ArrTy Arr;
	size_t Width;
	typedef typename ArrTy::iterator iterator;
	//static version
	FArray2d(size_t InWidth,size_t InHeight,ArrayType InitVal=ArrayType()):
		Arr(InWidth*InHeight,InitVal),
		Width(InWidth){}

	FArray2d(ArrayType InitVal=ArrayType()):
		FArray2d(0,0,InitVal){}

	size_t size()const{
		return Arr.size();
	}
    Point dim()const{
        return Point(Width,size()/Width);
    }

	void assign(ArrayType Val){
		for(ArrayType & v  : Arr){
			v = Val;
		}
	}
	iterator begin(){
		return Arr.begin();
	}
	iterator end(){
		return Arr.end();
	}
	ArrayType & operator [](Point P){
		return Arr[P.Y*Width + P.X];
	}
    ArrayType operator [](Point P)const{
		return Arr[P.Y*Width + P.X];
	}
	ArrayType * operator[](size_t Y){
		return &Arr[Y*Width];
	}
};
