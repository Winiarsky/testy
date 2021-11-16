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

if __name__ == '__main__':
    for test in (
        test_import_prime_factors,
        test_intiger_argument,
    ):  
        print(f'{test.__name__}: ', end='')
        try:
            test()
            print('OK')
        except AssertionError as error:
            print(error)
