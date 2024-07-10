# to show how to manually make
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include global/Tensor.cpp -o PyLRI$(python3-config --extension-suffix)