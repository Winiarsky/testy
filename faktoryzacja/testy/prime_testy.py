import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

def test_import_prime_factors():
    try:
        from prime import prime_factors  
    except ImportError as error:  
        assert False, error
        
def test_intiger_argument():
    try:
        from prime import prime_factors
        prime_factors(123)  
    except TypeError as error:  
        assert False, error

def test_return_list():
    from prime import prime_factors
    result = prime_factors(123)  
    assert isinstance(result,list), "Filed - result is not list"

def test_even_number_divisible():
    from prime import prime_factors
    result = prime_factors(40)  
    assert result == [2,2,2], "Filed - even number dvisible error"

def test_odd_case_factor():
    from prime import prime_factors
    result = prime_factors(40)  
    assert result == [2,2,2,5], "Filed - odd number dvisible error"


if __name__ == '__main__':
    for test in (
        test_import_prime_factors,
        test_intiger_argument,
        test_return_list,
        test_even_number_divisible,
        test_odd_case_factor,
    ):  
        print(f'{test.__name__}: ', end='')
        try:
            test()
            print('OK')
        except AssertionError as error:
            print(error)
