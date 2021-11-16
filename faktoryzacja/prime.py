import math
def prime_factors(number: int) -> list:
    if not isinstance(number, int):
        raise AttributeError
    
    if number == 1:
        return [1]

    factors = []
    #even number divisible
    while number % 2 == 0:
        factors.append(2)
        number = number / 2
    
    #n became odd
    for i in range(3,int(math.sqrt(number))+1,2):
     
      while (number % i == 0):
        factors.append(i)
        number = number / i
    
    if number > 2:
        factors.append(number)
        
    return factors

print(prime_factors(2))