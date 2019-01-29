#!/bin/bash

#activate an enviroment with acoular version to test
source activate acoular36
#build a test suite object which runs the tests in this folder
python -m unittest discover -v -p "testing*.py"


