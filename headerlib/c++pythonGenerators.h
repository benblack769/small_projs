//generators
#pragma once
#include <thread>
#include <atomic>
#define ShortSleep() std::this_thread::sleep_for(chrono::microseconds(1))
template<typename ReturnType>
class generator{
public:
	ReturnType CurVal;
	bool AtEnd;
	generator(void Fn(generator<ReturnType> *)){
		AtEnd = false;
		GenThread = std::thread(Fn,this);
		YeildLock.test_and_set();
		NextLock.clear();
	}
	generator(generator<ReturnType> &) = delete;
	void operator = (generator<ReturnType> &) = delete;

	void yeild(ReturnType & Val){
		while(!NextLock.test_and_set()) ShortSleep();//pauses until unlocked by yeild
		CurVal = Val;
		YeildLock.test_and_set();
		NextLock.clear();
	}
	void end(){
		AtEnd = true;
		NextLock.clear();
		YeildLock.test_and_set();
	}
	ReturnType next(){
		if (!AtEnd){
			YeildLock.clear();
			NextLock.test_and_set();
			while(!NextLock.test_and_set()) ShortSleep();//pauses until unlocked by yeild
		}
		return CurVal;
	}
protected:
	std::atomic_flag YeildLock, NextLock;
	std::thread GenThread;
private:
};
template<typename Type>
class GenIter{
public:
	class GenIterator{
	public:
		generator<Type> * Gen;
		Type CurVal;
		bool LastOne;
		GenIterator() = default;
		GenIterator(generator<Type> * Generator){
			Gen = Generator;
			CurVal = Gen->next();
		}
		bool operator != (GenIterator & Other){//argument unused!!!
			bool Return = LastOne;
			LastOne = !Gen->AtEnd;
			return Return;
		}
		void operator++(){
			CurVal = Gen->next();
		}
		Type operator *(){
			return CurVal;
		}
	};
	GenIterator Start,End;//End not initialized, null value!!!
	GenIter(generator<Type> * Generator){
		Start = GenIterator(Generator);
	};
	GenIterator & begin(){
		return Start;
	}
	GenIterator & end(){
		return End;
	}
};
template<typename YeildType>
GenIter<YeildType> Through(void GenFunc(generator<YeildType> *)){
	return GenIter<YeildType>(new generator<YeildType>(GenFunc));
}
