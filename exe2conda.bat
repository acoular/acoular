conda convert -d "numpy >=1.8" "scipy >=0.13" "scikit-learn >=0.15" "pytables >=3.1" "traits >=4.4" "chaco >=4.4" -o dist -v %1
if errorlevel 1 exit 1
conda create -y -n atest -c file:///%cd%/dist acoular
call activate atest
python -c "import acoular"
call deactivate
conda env remove -y -n atest