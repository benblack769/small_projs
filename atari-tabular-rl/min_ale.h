#pragma once
#include "ale_c_wrapper.h"

class MinALE{
    ALEInterface * state;
public:
    MinALE(){
        state = ALE_new();
    }
    ~MinALE(){
        ALE_del(state);
    }
    int act(int action){
        return ::act(state,action);
    }
    void loadROM(std::string s){ ::loadROM(state,s.c_str());}
    bool game_over(){return ::game_over(state);}
    void reset_game(){::reset_game(state);}
    int lives(){int x; ::lives(state,&x); return x;}
    int getScreenWidth(){return ::getScreenWidth(state);}
    int getScreenHeight(){return ::getScreenHeight(state);}
    void getScreenGrayscale(unsigned char *output_buffer){::getScreenGrayscale(state,output_buffer);}
    void setLoggerMode(int mode){::setLoggerMode(mode);}
};
