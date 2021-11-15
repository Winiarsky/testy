def test_import_prime_factors():
    try:
        from prime import prime_factors  
    except ImportError as error:  
        assert False, error


if __name__ == '__main__':
    for test in (
        test_import_prime_factors,
    ):  
        print(f'{test.__name__}: ', end='')
        try:
            test()
            print('OK')
        except AssertionError as error:
            print(error)