def poly(L, x):
    return sum(coef * (x ** i) for i, coef in enumerate(reversed(L)))

def poly_zeros(L, a, b):
    zeros = []
    for x in range(a, b + 1):
        if poly(L, x) == 0:
            if x not in zeros:
                zeros.append(x)
    return zeros

# Coefficients of the polynomial f(x) = x^6 - 4x^5 - 18x^4 + 52x^3 + 101x^2 - 144x - 180
coefficients = [1, -4, -18, 52, 101, -144, -180]

# Find zeros in the range [0, 4]
zeros_in_range = poly_zeros(coefficients, 0, 4)

# Output the number of integer zeros
print(len(zeros_in_range))  # This will give the number of integer zeros in the range [0, 4]
