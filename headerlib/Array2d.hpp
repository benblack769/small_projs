#pragma once
#include <array>
#include <vector>
#include "point.hpp"

template<typename ArrayType,size_t XSize,size_t YSize>
class Array2d
{
protected:
	static constexpr size_t ArrSize = XSize*YSize;
public:
	using ArrTy = std::array<ArrayType,ArrSize>;
	ArrTy Arr;
	typedef typename ArrTy::iterator iterator;
	//static version
	Array2d(ArrayType InitVal){
		assign(InitVal);
	}
	Array2d() = default;
	size_t size(){
		return Arr.size();
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
		return Arr[P.Y*YSize + P.X];
	}
	ArrayType * operator[](size_t Y){
		return &Arr[Y*YSize];
	}
};

template<typename ArrayType>
class FArray2d
{
public:
	using ArrTy = std::vector<ArrayType>;
	ArrTy Arr;
	size_t Height;
	typedef typename ArrTy::iterator iterator;
	//static version
	FArray2d(size_t InWidth,size_t InHeight,ArrayType InitVal=ArrayType()):
		Arr(InWidth*InHeight,InitVal),
		Height(InHeight){}

	FArray2d(ArrayType InitVal=ArrayType()):
		FArray2d(0,0,InitVal){}

	size_t size(){
		return Arr.size();
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
	ArrayType & operator [](Point & P){
		return Arr[P.Y*Height + P.X];
	}
	ArrayType * operator[](size_t Y){
		return &Arr[Y*Height];
	}
};
