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
    result = prime_factors(8)  
    assert result == [2,2,2], "Filed - even number dvisible error"

def test_odd_case_factor():
    from prime import prime_factors
    result = prime_factors(40)  
    assert result == [2,2,2,5], "Filed - odd number dvisible error"

def test_only_intiger_assertion():
    from prime import prime_factors
    try:
        assert isinstance(prime_factors(40.5), AttributeError), "Filed - not only intigers can be used"
    except AttributeError:
        return True

def test_one_case():
    from prime import prime_factors
    result = prime_factors(1)  
    assert result == [1], f"Filed - prime_factors(1) should be [1] got {result}"


if __name__ == '__main__':
    for test in (
        test_import_prime_factors,
        test_intiger_argument,
        test_return_list,
        test_even_number_divisible,
        test_odd_case_factor,
        test_only_intiger_assertion,
        test_one_case,
    ):  
        print(f'{test.__name__}: ', end='')
        try:
            test()
            print('OK')
        except AssertionError as error:
            print(error)
