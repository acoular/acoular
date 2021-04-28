#!/bin/bash

# building recipes for currently supported python versions
conda build --py 3.6 --no-copy-test-source-files recipe.local_py36
conda build --py 3.7 --no-copy-test-source-files recipe.local_py37
conda build --py 3.8 --no-copy-test-source-files recipe.local_py38
conda build --py 3.9 --no-copy-test-source-files recipe.local_py39

# convert built Linux packages for all OS
cd ~/anaconda3/conda-bld/linux-64/
conda convert -p all -o .. acoular*


