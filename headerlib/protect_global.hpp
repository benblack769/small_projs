#pragma once
/*
you can use this in the following way:

var global;//var is a type
void DoSomething(){
    auto Key = Protect(global);
    //change global
}//when Key destructs, it returns global to its previous value

it also allows for early restoring with Release, late protecting with
Protect, and switching keys that protect the global with
new = std::move(old)
*/
template<typename GlobType>
class ProtectedGlobal{
public:
    GlobType SavedGlobal;
    GlobType * Global = nullptr;
    ProtectedGlobal(){}
    ProtectedGlobal(GlobType & InGlob){
        Protect(InGlob);
    }
    ProtectedGlobal(ProtectedGlobal & Other) = delete;
    ProtectedGlobal(ProtectedGlobal && Other){
        *this = std::move(Other);
    }
    void operator = (ProtectedGlobal &) = delete;
    void operator = (ProtectedGlobal && Other){
        Global = Other.Global;
        SavedGlobal = Other.SavedGlobal;
        Other.Global = NULL;
    }
    ~ProtectedGlobal(){
        Release();
    }
    void Release(){
        if (Global != NULL){
            *Global = SavedGlobal;
            Global = NULL;
        }
    }
    void Protect(GlobType & InGlob){
        Release();
        SavedGlobal = InGlob;
        Global = &InGlob;
    }
};
template<typename GlobType>
inline ProtectedGlobal<GlobType> Protect(GlobType & InGlob){
    return ProtectedGlobal<GlobType>(InGlob);
}
