#!/bin/bash
g++ -O2 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` rod_decoder_py.cpp -o rod_decoder_py`python3-config --extension-suffix`
cp *.so ../../scripts/
