import numpy as np
from packages.utills import LinearProgrammingTabrizU as LP

# Entering the Target function
Z = [int(x) for x in input("Please enter coefficients of the target function: ").split()]
Z = np.array(Z)
variables_no = len(Z)

limit = int(input("Please enter the number of limitations that are needed: "))
A = []


# Entering the coefficients of limitations
counter = 0
while True:
    a = [int(x) for x in input("Please enter coefficients of the limitations: ").split()]
    if len(a) == variables_no:
        A.append(a)
        counter += 1
    else:
        print((f"Corresponding variables to assign coefficients are {variables_no}," + \
                " you are assiging the wrong amount of coefficients. try again"))
    
    if counter == limit + variables_no:
        A = np.array(A)
        break

B = []
for _ in range(len(A)):
    B.append(int(input("Please enter the boundries: ")))
B = np.array(B)


solver = LP(Z, A, B, variables_no)
print(solver.find_optimal_extreme_points())
print(solver.find_recession_directions())
solver.finite_optimal_solutions()

