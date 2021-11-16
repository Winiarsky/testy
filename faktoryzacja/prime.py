def prime_factors(number: int) -> list:
    factors = []
    #even number divisible
    while number % 2 == 0:
        factors.append(2)
        number = number / 2
    return factors