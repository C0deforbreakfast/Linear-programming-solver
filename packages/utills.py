import itertools as it
import numpy as np


class LinearProgrammingTabrizU:
    def __init__(self, target_func_coefficient: np.ndarray, constraint_coefficients: np.ndarray, constants: np.ndarray, variable_num: int, show_stat=False) -> None:
        self.show_stat = show_stat
        self.target_func_coefficient = target_func_coefficient
        self.constraint_coefficients = constraint_coefficients
        self.constants = constants
        self.variable_num = variable_num

        '''
            Take a copy of constraints coefficients and constants for finding optimal extreme points.
            Add d1 + d2 + ... + dn = 1 to both constraints and constants for calculating recession directions.
        '''
        self.constraint_coefficients_copy = self.constraint_coefficients.copy()
        self.constraint_coefficients = np.append(self.constraint_coefficients, np.array([[1] * len(self.constraint_coefficients[0])]), axis=0)

        self.constants_copy = self.constants.copy()
        self.constants.fill(0)
        self.constants = np.append(self.constants, [1], axis=0)

        self.optimal_extreme_points = []
        self.recession_directions = []

    def stats(self, value, end=False):
        '''
            A functionality for printing stats.
            fyi prints calculations
        '''
        if self.show_stat and not end:
            print(value)
            print("-----------------")
        elif self.show_stat and end:
            print(value, end=" ")

    def find_optimal_extreme_points(self):
        combination = it.combinations(range(len(self.constraint_coefficients_copy)), self.variable_num)

        for comb in combination:
            '''
                When we are solving the problem for n variables we have n combinations of constraints and their corresponding constants
            '''
            constraints = [self.constraint_coefficients_copy[i].tolist() for i in comb]
            constants = [self.constants_copy[i].tolist() for i in comb]

            if np.linalg.det(constraints) != 0:
                solution = np.linalg.solve(constraints, constants).reshape(self.variable_num, 1)
                self.stats(f"A.x = b ==>{constraints}.{constants} = {solution}", end=True)

                for i in range(len(self.constraint_coefficients_copy)):
                    '''
                        We check the solution with every constraints we have for feasible solutions
                    '''
                    if self.constraint_coefficients_copy[i] @ solution > self.constants_copy[i]:
                        self.stats(f"{self.constraint_coefficients_copy[i]} @ {solution} > {self.constants_copy[i]}, feasible: {bool(self.constraint_coefficients_copy[i] @ solution > self.constants_copy[i])}")
                        break
                else:
                    self.stats(f"{self.constraint_coefficients_copy[i]} @ {solution} > {self.constants_copy[i]}, feasible: {bool(self.constraint_coefficients_copy[i] @ solution > self.constants_copy[i])}")
                    self.optimal_extreme_points.append(solution)
        
        return f"OPTIMAL EXTREME POINTS: \n {self.optimal_extreme_points}"

    def find_recession_directions(self):
        combination = it.combinations(range(len(self.constraint_coefficients)), self.variable_num)

        for comb in combination:
            '''
                When we are solving the problem for n variables we have n combinations of constraints and their corresponding constants
            '''
            constraints = [self.constraint_coefficients[i].tolist() for i in comb]
            constants = [self.constants[i].tolist() for i in comb]

            if np.linalg.det(constraints) != 0:
                '''
                    Solves the equation and reshapes the solution to dimensions of (variable_num X 1)
                '''
                recession_direction = np.linalg.solve(constraints, constants).reshape(self.variable_num, 1)
                self.stats(f"A.d = b ==>{constraints}.{constants} = {recession_direction}", end=True)

            for i in range(len(self.constraint_coefficients)):
                if [1, 1] @ recession_direction != 1:
                    self.stats(f"{[1, 1]} @ {recession_direction} != 1, feasible: {bool([1, 1] @ recession_direction != 1)}")
                    break
                if self.constraint_coefficients[i] @ recession_direction > self.constants[i]:
                    self.stats(f"{self.constraint_coefficients[i]} @ {recession_direction} > {self.constants[i]}, feasible: {bool(self.constraint_coefficients[i] @ recession_direction > self.constants[i])}")
                    break
            else:
                self.stats(f"{self.constraint_coefficients[i]} @ {recession_direction} > {self.constants[i]}, feasible: {bool(self.constraint_coefficients[i] @ recession_direction > self.constants[i])}")
                self.recession_directions.append(recession_direction)    

        return f"RECESSION_DIRECTIONS: \n {self.recession_directions}"

    def finite_optimal_solutions(self):
        feasible_solutions = []
        
        for rd in self.recession_directions:
            '''
                Does dot products with each target function coefficients and recession directions and if it's less than or equal to zero
                then it means the angle between two vectors are not acute and the problem doesn't have finite optimal solutions
            '''
            if self.target_func_coefficient @ rd <= 0:
                self.stats(f"The angle between vector {self.target_func_coefficient} and recession direction {rd} is not an acute angle so")
                print("The problem has no finite optimal solutions!")
                break
        else:
            '''
                If all the recession directions have acute angles with target function coefficients, then problem has finite optimal solutions
                and to calculate we simply do the dot product of target function coefficients with extreme optimal points
            '''
            for eop in self.optimal_extreme_points:
                self.stats(f"{self.target_func_coefficient}.{eop} = {self.target_func_coefficient @ eop}")
                feasible_solutions.append(self.target_func_coefficient @ eop)
        
            print(f"All the feasible solutions: {feasible_solutions}")
            print(f"Minimum solution: {min(feasible_solutions)}")


