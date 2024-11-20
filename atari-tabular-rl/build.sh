ALE_LIB_DIR=$(python -c "import multi_agent_ale_py; import os; print(os.path.dirname(multi_agent_ale_py.__file__));")
echo $ALE_LIB_DIR
# cp  $ALE_LIB_DIR/.libs/* .
# cp  $ALE_LIB_DIR/libale_c.so .
#export LD_LIBRARY_DIR=$LD_LIBRARY_DIR:$ALE_LIB_DIR/../../
g++ -g -O3 -Wall -shared -std=c++11 -fPIC  main.cpp md5.cpp `python3 -m pybind11 --includes` -L $ALE_LIB_DIR -l:libale_c.so -o example`python3-config --extension-suffix` #-Wl,-rpath-link='$ORIGIN',-rpath-link=/home/benblack/anaconda3/lib/  -L . -L $ALE_LIB_DIR/../../ -l:libale_c.so
