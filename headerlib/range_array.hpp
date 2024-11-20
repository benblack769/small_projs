/*
Two dimentional array with offset.

Allows one to access points efficiently in a  two dimentional space with
arbirary bounds.

So you can have a space like([-10,10],[-20,0]), and easily and efficiently store data in it.
*/
#pragma once
#include <headerlib/point.hpp>
#include <vector>
#include <exception>

template<typename ArrayType,typename ContTy=std::vector<ArrayType>>
class RangeArray;

template<typename ArrayType,typename ContTy>
class RA_Iterator{
public:
    PointIter Spot;
    using ArrIterator = typename ContTy::iterator;
    ArrIterator ArrIt;
    RA_Iterator(RangeArray<ArrayType> * InArr){
        ArrIt = InArr->Arr.begin();

        int sx = InArr->Corner.X;
        int sy = InArr->Corner.Y;
        int xdis = InArr->XSize;
        int ydis = InArr->YSize;
        Spot = PointIter(sx, sy, sx + xdis, sy + ydis);
    }
    RA_Iterator(){
        Spot = PointIter();
    }
    bool operator != (RA_Iterator & Other){
        return Spot.NotEnd();
    }
    void operator ++ (){
        ++Spot;
        ++ArrIt;
    }
    PointInfo<ArrayType> operator *(){
        return PointInfo<ArrayType>(*Spot, &(*ArrIt));
    }
};

template<typename ArrayType,typename ContTy>
class RangeArray
{
public:
    ContTy Arr;
    using iterator = RA_Iterator<ArrayType,ContTy>;
    Point Corner;
    int YSize, XSize;
    RangeArray(Point offset,int xsize,int ysize,const ContTy & cont):
        Arr(cont),
        Corner(offset),
        YSize(ysize),
        XSize(xsize){}
    
    RangeArray(){
        Corner = Point(0, 0);
        YSize = 0;
        XSize = 0;
    }
    RangeArray(const RangeArray & other){
        (*this) = other;
    }
    RangeArray(RangeArray && other){
        (*this) = other;
    }

    void operator =(const RangeArray & val){
        _copy_scalars(val);
        Arr = val.Arr;
    }
    void operator =(RangeArray && val){
        _copy_scalars(val);
        Arr = move(val.Arr);
    }
    void assign(ArrayType InitVal){
        for (auto & Val : Arr)
            Val = InitVal;
    }

    size_t size(){
        return XSize*YSize;
    }
    size_t PointArrIdx(Point P){
        return (P.Y - Corner.Y) * XSize + (P.X - Corner.X);
    }
    iterator begin(){
        return iterator(this);
    }
    iterator end(){
        return iterator();
    }
    bool IsInScope(Point P){
        int Xadj = P.X - Corner.X;
        int Yadj = P.Y - Corner.Y;
        return Xadj < XSize && Yadj < YSize && Xadj >= 0 && Yadj >= 0;
    }
    decltype (Arr[0]) operator [](Point P){
        return Arr[PointArrIdx(P)];
    }
    decltype (Arr[0]) at(Point P){
        if(!IsInScope(P)){
            throw std::runtime_error("not in RangeArray scope");
        }
        return (*this)[P];
    }

protected:
    void _copy_scalars(const RangeArray & other){
        Corner = other.Corner;
        YSize = other.YSize;
        XSize = other.XSize;
    }
};
template<typename data_ty>
bool SameScope(const RangeArray<data_ty> & ra1,const RangeArray<data_ty> & ra2){
    return ra1.Corner == ra2.Corner &&
            ra1.XSize == ra2.XSize &&
            ra1.YSize == ra2.YSize;
}
