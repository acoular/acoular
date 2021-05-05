#!/bin/bash

#remove cache data before testing
rm -rf ./cache/*
rm -rf ./__pycache__

#build a test suite object which runs the tests in this folder
python -m unittest discover -v -p "test_*.py"

VAL=$?

#remove cache data after testing
rm -rf ./cache/*
rm -rf ./__pycache__

exit $VAL


