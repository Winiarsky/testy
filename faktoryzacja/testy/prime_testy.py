# def test_import_prime_factors():
#     try:
#         from prime import prime_factors  
#     except ImportError as error:  
#         assert False, error

# # def test_intiger_argument():
# #     try:
# #         from faktoryzacja.prime import prime_factors  
# #         prime_factors(123)
# #     except ImportError as error:  
# #         assert False, error


# if __name__ == '__main__':
#     for test in (
#         test_import_prime_factors,
#         # test_intiger_argument,
#     ):  
#         print(f'{test.__name__}: ', end='')
#         try:
#             test()
#             print('OK')
#         except AssertionError as error:
#             print(error)

import sys
import os
  
# setting path
# sys.path.append('../faktoryzacja')
print(os.path.isdir('./faktoryzacja/prime.py'))
