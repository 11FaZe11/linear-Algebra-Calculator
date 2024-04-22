import time

while True:
  try:
    print("""  
1- Addition of matrixs.
2- Subtraction of matrixs.
3- Multiplication of 2 matrix.
4- Guass-Jordon elimination to get unknowns.
5- Guass-Jordon elimination to get the inverse.
6- Get the Det of matrix.
7- Get the Transpose of matrix.
8- Get the Dot product of 2 vectors.
9- Get the Cross product of 2 vectors.
10- Get the GCD of 2 numbers.
0- End the program.
""")

    first_choice = int(input("\nEnter the operation you wish..... \n"))


    if first_choice in range(0,11):

        if first_choice == 1:
            def add_matrices(matrix1, *matrices):
                n = len(matrix1)
                m = len(matrix1[0])

                if not all(len(matrix) == n and len(matrix[0]) == m for matrix in matrices):
                    return None

                sum_matrix = [[0 for _ in range(m)] for _ in range(n)]

                for i in range(n):
                    for j in range(m):
                        sum_matrix[i][j] = matrix1[i][j] + sum(matrix[i][j] for matrix in matrices)

                return sum_matrix


            n = int(input("Enter the number of rows and columns for the matrices: "))

            matrix1 = []
            for i in range(n):
                print("Enter elements of row ", i + 1, " for the first matrix: ", sep="")
                print("HINT::How to enter the input ---> Use SPACE to separate the numbers ----> Example 1 2 3.")
                row = [float(x) for x in input().split()]
                matrix1.append(row)

            matrices = []
            while True:
                more_matrices = input("Enter 'yes' to add another matrix, or anything else to proceed: ")
                if more_matrices.lower() != 'yes':
                    break

                matrix = []
                for i in range(n):
                    print("Enter elements of row", i + 1, "for the additional matrix: ", sep="")
                    row = [float(x) for x in input().split()]
                    matrix.append(row)
                matrices.append(matrix)

            sum_matrix = add_matrices(matrix1, *matrices)

            if sum_matrix is None:
                print("Matrices have different dimensions. Addition not possible.")
            else:
                print("Sum of the matrices:")
                for row in sum_matrix:
                    print(*row)


        elif first_choice == 2:
            def subtract_matrices(matrix1, *matrices):

                n = len(matrix1)
                m = len(matrix1[0])

                if not all(len(matrix) == n and len(matrix[0]) == m for matrix in matrices):
                    return None

                difference_matrix = [[0 for _ in range(m)] for _ in range(n)]

                for i in range(n):
                    for j in range(m):
                        difference_matrix[i][j] = matrix1[i][j] - sum(matrix[i][j] for matrix in matrices)

                return difference_matrix


            n = int(input("Enter the number of rows and columns for the matrices: "))

            matrix1 = []
            for i in range(n):
                print("Enter elements of row ", i + 1, " for the first matrix: ", sep="")
                print("HINT::How to enter the input ---> Use SPACE to separate the numbers ----> Example 1 2 3.")
                row = [float(x) for x in input().split()]
                matrix1.append(row)

            matrices = []
            while True:
                more_matrices = input("Enter 'yes' to add another matrix, or anything else to proceed: ")
                if more_matrices.lower() != 'yes':
                    break

                matrix = []
                for i in range(n):
                    print("Enter elements of row", i + 1, "for the additional matrix: ", sep="")
                    row = [float(x) for x in input().split()]
                    matrix.append(row)
                matrices.append(matrix)

            difference_matrix = subtract_matrices(matrix1, *matrices)

            if difference_matrix is None:
                print("Matrices have different dimensions. Subtraction not possible.")
            else:
                print("Difference of the matrices:")
                for row in difference_matrix:
                    print(*row)


        elif first_choice == 3:
            def multiply_matrices(matrix1, matrix2):
                n = len(matrix1)
                m = len(matrix1[0])
                p = len(matrix2[0])

                if m != len(matrix2):
                    return None

                product_matrix = [[0 for _ in range(p)] for _ in range(n)]

                for i in range(n):
                    for j in range(p):
                        for k in range(m):
                            product_matrix[i][j] += matrix1[i][k] * matrix2[k][j]

                return product_matrix


            def get_matrix(rows, cols):
                matrix = []
                for i in range(rows):
                    row = []
                    for j in range(cols):
                        value = float(input(f"Enter element at row {i + 1}, column {j + 1}: "))
                        row.append(value)
                    matrix.append(row)
                return matrix


            n1 = int(input("Enter the number of rows for the first matrix: "))
            m1 = int(input("Enter the number of columns for the first matrix: "))

            n2 = int(input("Enter the number of rows for the second matrix: "))
            m2 = int(input("Enter the number of columns for the second matrix: "))

            if m1 != n2:
                print("Incompatible dimensions for multiplication. Exiting.")
            else:
                matrix1 = get_matrix(n1, m1)
                print("First matrix entered successfully!")

                matrix2 = get_matrix(n2, m2)

                product_matrix = multiply_matrices(matrix1, matrix2)

                if product_matrix is None:
                    print("Error: Incompatible matrix dimensions.")
                else:
                    print("Product of the matrices:")
                    for row in product_matrix:
                        print(*row)


        elif first_choice == 4:
            def gauss_jordan(A, b):

                n = len(A)

                Ab = [[A[i][j] for j in range(n)] + [b[i]] for i in range(n)]

                for i in range(n):
                    if Ab[i][i] == 0:

                        all_zero = True
                        for j in range(i + 1, n):
                            if any(Ab[j][k] != 0 for k in range(n + 1)):
                                all_zero = False
                                break
                        if all_zero:
                            return None

                        for j in range(i + 1, n):
                            if Ab[j][i] != 0:
                                Ab[i], Ab[j] = Ab[j], Ab[i]
                                break
                        if Ab[i][i] == 0:
                            return None

                    for j in range(i + 1, n):
                        factor = Ab[j][i] / Ab[i][i]
                        for k in range(n + 1):
                            Ab[j][k] -= factor * Ab[i][k]

                x = [0] * n
                for i in range(n - 1, -1, -1):
                    sum = 0
                    for j in range(i + 1, n):
                        sum += Ab[i][j] * x[j]
                    x[i] = (Ab[i][n] - sum) / Ab[i][i]

                return x


            n = int(input("""Enter the number of equations (matrix size): 
HINT::How to enter the input ---> Use SINGE value.--It should be a square matrix.---> Example 1 or 2 or 3
"""))

            A = []
            for i in range(n):
                print("HINT::How to enter the input ---> Use SPACE to separate the numbers ----> Example 1 2 3.")
                print("Enter elements of row ", i + 1, ": ", sep="")

                row = [float(x) for x in input().split()]
                A.append(row)

            b = []
            print("HINT::How to enter the input ---> Use SPACE to separate the numbers ----> Example 1 2 3.")
            print("Enter the constants vector (right-hand side): ", sep="")
            b = [float(x) for x in input().split()]

            solution = gauss_jordan(A, b)

            if solution is None:
                print("Sorry but no solution exists")
            else:
                print("Solution:", solution)


        elif first_choice == 5:
            n = int(input("""Enter no of rows of the square matrix: 
HINT::How to enter the input ---> Use SINGE value.--It should be a square matrix.---> Example 1 or 2 or 3
"""))
            matrix = [
                [int(input("Enter element " + str(j + 1) + " , " + str(i + 1) + " of the matrix: ")) for i in range(n)]
                for
                j in
                range(n)]


            def gen_id_mat(n):
                l = [[0 for i in range(n)] for j in range(n)]
                for k in range(n):
                    l[k][k] = 1
                return l


            def gen_id_mat1(n):
                l = [[0.0 for i in range(n)] for j in range(n)]
                for k in range(n):
                    l[k][k] = 1.0
                return l


            def row_swap(m, r1, r2):
                m[r1], m[r2] = m[r2], m[r1]
                return m


            def row_op_1(m, r1, r2, c):
                for i in range(len(m)):
                    m[r1][i] = (m[r2][i]) * c
                return m


            def row_op_2(m, r1, r2, c):
                for i in range(len(m)):
                    m[r1][i] = m[r1][i] - (c * m[r2][i])
                return m


            def disp(m):
                print('\n'.join([' '.join(['{:4}'.format(item) for item in row]) for row in m]))
                print()


            idm = gen_id_mat(n)
            id_inv = gen_id_mat(n)

            count = 0
            for col in range(n):
                for row in range(n):
                    if idm[row][col] == 1 and matrix[row][col] == 0:
                        for g in range(n):
                            if matrix[g][col] != 0:
                                matrix = row_swap(matrix, row, g)
                                print()
                                disp(matrix)
                                print("    ", "R" + str(row + 1) + " <--> " + "R" + str(g + 1))
                                print()
                    if matrix[row][col] != 0 and idm[row][col] == 1:
                        multp = 1 / matrix[row][col]
                        id_inv = row_op_1(id_inv, row, row, (1 / matrix[row][col]))
                        matrix = row_op_1(matrix, row, row, (1 / matrix[row][col]))
                        print()
                        count += 1
                        print("Step " + str(count) + ":")
                        print()
                        disp(matrix)
                        disp(id_inv)
                        print("    ", "R" + str(row + 1) + " --> " + "R" + str(row + 1) + " x " + str(multp))
                        print()

                        for const in range(n):
                            if const == row:
                                continue
                            multp = matrix[const][col]
                            id_inv = row_op_2(id_inv, const, row, matrix[const][col])
                            matrix = row_op_2(matrix, const, row, matrix[const][col])

                            count += 1
                            print("Step " + str(count) + ":")
                            print()
                            disp(matrix)
                            disp(id_inv)
                            print("    ",
                                  "R" + str(row + 1) + " --> " + "R" + str(row + 1) + " - " + str(
                                      multp) + " x " + "R" + str(
                                      const + 1))
                            print()

            if matrix == gen_id_mat1(n):
                print("The inverse of the given matrix is:")
                print()
                disp(id_inv)
            else:
                print("This matrix is not invertible...")


        elif first_choice == 6:
            def multiply_matrices(matrix1, matrix2):
                n = len(matrix1)
                m = len(matrix1[0])
                p = len(matrix2[0])

                if m != len(matrix2):
                    return None

                product_matrix = [[0 for _ in range(p)] for _ in range(n)]

                for i in range(n):
                    for j in range(p):
                        for k in range(m):
                            product_matrix[i][j] += matrix1[i][k] * matrix2[k][j]

                return product_matrix


            def get_matrix(rows, cols):
                matrix = []
                for i in range(rows):
                    row = []
                    for j in range(cols):
                        value = float(input(f"Enter element at row {i + 1}, column {j + 1}: "))
                        row.append(value)
                    matrix.append(row)
                return matrix


            def determinant(matrix):

                if len(matrix) != len(matrix[0]):
                    return None

                if len(matrix) == 1:
                    return matrix[0][0]
                elif len(matrix) == 2:
                    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

                determinant_sum = 0
                for i in range(len(matrix)):
                    sub_matrix = [row[:i] + row[i + 1:] for row in matrix[1:]]
                    determinant_sum += matrix[0][i] * determinant(sub_matrix) * (-1) ** i

                return determinant_sum


            n = int(input("""Enter the dimension of the square matrix: 
HINT::How to enter the input ---> Use SINGE value.--It should be a square matrix.---> Example 1 or 2 or 3            
"""))

            matrix = get_matrix(n, n)

            det = determinant(matrix)

            if det is None:
                print("Error: Invalid matrix dimensions.")
            else:
                print("\nDeterminant of the matrix:", det)


        elif first_choice == 7:
            def get_matrix(rows, cols):
                matrix = []
                for i in range(rows):
                    row = []
                    for j in range(cols):
                        value = float(input(f"Enter element at row {i + 1}, column {j + 1}: "))
                        row.append(value)
                    matrix.append(row)
                return matrix


            def transpose_matrix(matrix):

                rows = len(matrix)
                cols = len(matrix[0])

                transposed_matrix = [[0 for _ in range(rows)] for _ in range(cols)]

                for i in range(rows):
                    for j in range(cols):
                        transposed_matrix[j][i] = matrix[i][j]

                return transposed_matrix


            n = int(input("Enter the number of rows for the matrix: "))
            m = int(input("Enter the number of columns for the matrix: "))

            matrix = get_matrix(n, m)

            transposed_matrix = transpose_matrix(matrix)

            print("\nTranspose of the matrix:\n ")
            for row in transposed_matrix:
                print(*row)


        elif first_choice == 8:
            def get_vector_valu():

                vector = []
                num_elements = int(input("Enter the number of elements in the vector: "))
                for i in range(num_elements):
                    value = float(input(f"Enter element {i + 1} of the vector: "))
                    vector.append(value)
                return vector


            # retrun in form of array

            def dot_product(vector1, vector2):
                # the most imporat func in the program
                # calculate the dot prod of 2 V

                if len(vector1) != len(vector2):
                    raise ValueError("Error: Incompatible vector dimensions for dot product.")

                dot_product_sum = 0
                for i in range(len(vector1)):
                    dot_product_sum += vector1[i] * vector2[i]

                return dot_product_sum


            # retrun the valu in form of array

            print("Enter the elements of the first vector:")
            vector1 = get_vector_valu()

            print("Enter the elements of the second vector:")
            vector2 = get_vector_valu()

            try:
                dot_product_result = dot_product(vector1, vector2)
                print("\nDot product of the vectors:", dot_product_result)
            except ValueError as e:
                print(e)
            # stolen from jonior_nova XD


        elif first_choice == 9:
            def get_vector():
                """
                This function prompts the user to enter the dimension and components of a vector and returns it as a list.
                """
                while True:
                    try:
                        dimension = int(input("Enter the dimension of the vector (2 or 3): "))
                        if dimension not in (2, 3):
                            raise ValueError("Invalid dimension. Please enter 2 or 3.")
                        break
                    except ValueError as e:
                        print("Error:", e)

                components = []
                for i in range(dimension):
                    component = float(input(f"Enter component {i + 1} of the vector: "))
                    components.append(component)
                return components


            def cross_product(a, b):
                """
                This function computes the cross product of two vectors a and b (represented as lists).

                Args:
                    a: A list representing the first vector.
                    b: A list representing the second vector.

                Returns:
                    A list representing the cross product of a and b, or None if dimensions are incompatible.
                """

                dim_a = len(a)
                dim_b = len(b)

                # Handle compatible dimensions (2D or 3D cross product)
                if dim_a == 2 and dim_b == 2:
                    return [0, 0, a[0] * b[1] - a[1] * b[0]]  # 2D cross product (z-component)
                elif dim_a == 3 and dim_b == 3:
                    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
                else:
                    # Handle incompatible dimensions
                    return None


            # Get vector components from the user
            vector1 = get_vector()
            vector2 = get_vector()

            result = cross_product(vector1, vector2)

            if result is not None:
                print("The cross product of the vectors is:", result)
            else:
                print("Error: Incompatible vector dimensions for cross product.")


        elif first_choice == 10:
            def gcd(a, b):

                print(f"Step 1: Initial values - a = {a}, b = {b}")
                while b != 0:
                    remainder = a % b
                    print(f"Step 2: Remainder - a = {a}, b = {b}, remainder = {remainder}")
                    a = b
                    b = remainder
                    print(f"Step 3: Update values - a = {a}, b = {b}")
                    print("_")
                return a


            while True:
                try:
                    num1 = int(input("Enter the first number: "))
                    num2 = int(input("Enter the second number: "))
                    break
                except ValueError:
                    print("Invalid input. Please enter integers only.")

            gcd_result = gcd(num1, num2)
            print(f"Step 4: GCD - The greatest common divisor of {num1} and {num2} is: {gcd_result}")


        elif first_choice == 0:
            break

    else:
      print("Invalid input. Please enter from the previos choices")
    print("""
        the program will start in 3 seconds
        """)
    for abc in range(1,4):
        print(abc)
        time.sleep(1)

  except ValueError:

    print("Invalid input. Please enter a number.")



  if first_choice in range(0,1000000000000000000000000000000000000000000000000) and first_choice != 0:
    print("Do you want to enter another option? (yes/no)")

    answer = input().lower()
    if answer != 'yes':
      break
print("Thank you for using this tool !!")


