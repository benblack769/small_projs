#include "mainwindow.h"
#include <QApplication>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <ctime>

int main(int argc, char *argv[])
{
    srand(clock());
    QApplication a(argc, argv);
    
    assert(argc==7 &&  "wrong number of arguments");
    string infilename = argv[1];
    is_lin_rand_walk = string(argv[2]) == "true";
    rand_lin_val = stod(string(argv[3]));
    dest_lin_val  =  stod(string(argv[4]));
    avoid_prev_lin_val  =  stod(string(argv[5]));
    walk_dis =  stoi(string(argv[6]));
    cout << is_lin_rand_walk << "\t"
         << rand_lin_val << "\t"
         << dest_lin_val << "\t" 
         << avoid_prev_lin_val << "\t" 
         << walk_dis << "\t";
            
    MainWindow w(infilename);
    w.show();
    
    return 0;
}
