import pytest

#%% fixtures provide isolation between tests!
# * return values from fixtures are cached!
# * fixtures can be found in conftest.py

some_object = []

@pytest.fixture
def add_element():
    some_object.append(len(some_object)+1)
    return some_object

def test_add_element(add_element):
    assert add_element == [1]

def test_add_element2(add_element):
    assert add_element == [1,2]

#%% 
import numpy as np 
import time

@pytest.fixture(scope='session')
#@pytest.fixture
def curent_time(): # in milliseconds
    return time.strftime("%H:%M:%S", time.localtime())

def test_current_time(curent_time):
    assert curent_time == False

def test_current_time2(curent_time):
    time.sleep(1)
    assert curent_time == False


