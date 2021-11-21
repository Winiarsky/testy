import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from prime import prime_factors

def test_intiger_argument():
    with pytest.raises(AttributeError):
        prime_factors("123")  

def test_return_list():
    result = prime_factors(123)  
    assert isinstance(result,list)

def test_even_number_divisible():
    result = prime_factors(8)  
    assert result == [2,2,2]

def test_odd_case_factor():
    result = prime_factors(40)  
    assert result == [2,2,2,5]

def test_only_intiger_assertion():
    with pytest.raises(AttributeError):
        prime_factors(40.5)  

def test_one_case():
    result = prime_factors(1)  
    assert result == [1]

def test_only_positive_intiger_assertion():
    with pytest.raises(AttributeError):
        prime_factors(-1)