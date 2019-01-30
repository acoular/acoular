#!/bin/bash

#activate an enviroment with the acoular version to test
source activate acoular36
#remove cache data before testing
rm -rf ./cache/*
#build a test suite object which runs the tests in this folder
python -m unittest discover -v -p "testing*.py"


