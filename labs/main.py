from system_create import (create_input_system, generate_random_matrix, generate_diagonally_dominant_matrix,
                           generate_random_vector, generate_spd_matrix, generate_symmetric_matrix)
from gauss_method import gauss_method
from optimal_exclusion import optimal_exclusion
from cholesky_method import cholesky_method
from lu_decomposition import lu_decomposition
from bordering_method import bordering_method
from qr_decomposition import qr_decomposition
from jacobi_method import jacobi_method, check_jacobi_convergence
from gradient_method import gradient_descent, check_gradient_descent_convergence
from sor_method import sor_method, check_sor_convergence
from rotation_with_barriers import rotation_with_barriers
from richardson_method import richardson_method
from inverse_iteration import inverse_iteration
from utils import is_singular, print_system, execute_method, is_symmetrical, execute_eigen_method, is_spd, execute_partial_eigen_method


def main_menu():
    while True:
        print("\nМЕНЮ")
        print("="*50)
        print("1. Создать систему")
        print("2. Уйти")
        print("="*50)

        choice = input("\nВыбор: ").strip()

        if choice == "1":
            create_system_menu()
        elif choice == "2":
            exit()
        else:
            print("\n(!) Некорректный ввод")


def create_system_menu():
    while True:
        print("\nСОЗДАНИЕ СИСТЕМЫ")
        print("="*50)
        print("1. Случайная генерация")
        print("2. Случайная генерация (симметричная матрица)")
        print("3. Случайная генерация (симметричная положительно определённая матрица)")
        print("4. Случайная генерация (матрица со строгим диагональным преобладанием)")
        print("5. Ручной ввод")
        print("6. Назад")
        print("="*50)
        
        choice = input("\nВыбор: ").strip()
        
        if choice == "1":
            random_system_flow()
        elif choice == "2":
            random_system_flow(symmetrical=True)
        elif choice == "3":
            random_system_flow(spd=True)
        elif choice == "4":
            random_system_flow(diagonally_dominant=True)
        elif choice == "5":
            input_system_flow()
        elif choice == "6":
            return
        else:
            print("\n(!) Некорректный ввод")


def random_system_flow(symmetrical=False, spd=False, diagonally_dominant=False):
    print("\nСЛУЧАЙНАЯ ГЕНЕРАЦИЯ")
    print("="*50)
    
    while True:
        try:
            n = int(input("Введите размерность системы (n): "))
            if n <= 0:
                print("(!) Размерность должна быть положительной")
                continue
            break
        except ValueError:
            print("(!) Введите целое положительное число")
    
    while True:
        try:
            d = int(input("Введите разброс значений d (диапазон [-d, d]): "))
            if d <= 0:
                print("(!) Разброс должен быть положительным")
                continue
            break
        except ValueError:
            print("(!) Введите целое положительное число")
    
    b = generate_random_vector(n, d)
    if symmetrical:
        A = generate_symmetric_matrix(n, d)
    elif spd:
        A = generate_spd_matrix(n, d)
    elif diagonally_dominant:
        A = generate_diagonally_dominant_matrix(n, d)
    else:
        A = generate_random_matrix(n, d)

    print()
    print_system(A, b)
    
    choose_method_menu(A, b)


def input_system_flow():
    print("\nРУЧНОЙ ВВОД")
    print("="*50)
    
    while True:
        try:
            n = int(input("Введите размерность системы (n): "))
            if n <= 0:
                print("(!) Размерность должна быть положительной")
                continue
            break
        except ValueError:
            print("(!) Введите целое положительное число")
    
    A, b = create_input_system(n)
    print_system(A, b)
    
    choose_method_menu(A, b)


def choose_method_menu(A, b):
    while True:
        print("\nВЫБОР МЕТОДА")
        print("="*50)
        print("1. Назад")
        print("----- РЕШЕНИЕ СИСТЕМЫ -----")
        print("2. Метод Гаусса с поиском максимального по матрице")
        print("3. Метод оптимального исключения")
        print("4. Метод квадратного корня (метод Холецкого)")
        print("5. Метод LU-разложения")
        print("6. Метод окаймления")
        print("7. Метод отражений (QR-разложение)")
        print("8. Метод Якоби")
        print("9. Метод последовательной релаксации")
        print("10. Метод наискорейшего градиентного спуска")
        print("11. Метод Ричардсона")
        print("----- НАХОЖДЕНИЕ СОБСТВЕННЫХ ЗНАЧЕНИЙ -----")
        print("12. Метод вращения с преградами")
        print("13. Метод обратной итерации")
        print("="*50)
        
        choice = input("\nВыбор: ").strip()
        
        if choice == "1":
            return
        elif choice == "2":
            if is_singular(A):
                print("(!) Матрица вырождена")
                continue
            execute_method(A, b, gauss_method)    
        elif choice == "3":
            if is_singular(A):
                print("(!) Матрица вырождена")
                continue
            execute_method(A, b, optimal_exclusion)
        elif choice == "4":
            if is_singular(A):
                print("(!) Матрица вырождена")
                continue
            if not is_symmetrical(A):
                print("(!) Матрица не симметрична")
                continue
            execute_method(A, b, cholesky_method) 
        elif choice == "5":
            if is_singular(A):
                print("(!) Матрица вырождена")
                continue
            execute_method(A, b, lu_decomposition)
        elif choice == "6":
            if is_singular(A):
                print("(!) Матрица вырождена")
                continue
            execute_method(A, b, bordering_method)
        elif choice == "7":
            execute_method(A, b, qr_decomposition)
        elif choice == "8":
            if not check_jacobi_convergence(A):
                print("(!) Метод Якоби не сходится по критерию сходимости")
                continue
            execute_method(A, b, jacobi_method)
        elif choice == "9":
            omega = float(input("Введите параметр омега: "))
            if not check_sor_convergence(A, omega):
                print("(!) Метод последовательной релаксации не сходится по критерию сходимости")
                continue
            execute_method(A, b, sor_method, omega=omega)
        elif choice == "10":
            if not check_gradient_descent_convergence(A):
                print("(!) Метод градиентного спуска не сходится по критерию сходимости")
                continue
            execute_method(A, b, gradient_descent)
        elif choice == "11":
            if not is_spd(A):
                print("(!) Метод Ричардсона не сходится т.к. матрица не является симметричной положительно определенной")
                continue
            execute_method(A, b, richardson_method)
        elif choice == "12":
            if not is_symmetrical(A):
                print("(!) Матрица несимметрична")
                continue
            execute_eigen_method(A, rotation_with_barriers)
        elif choice == "13":
            if not is_spd(A):
                print("(!) Матрица не является симметричной положительно определенной")
                continue
            execute_partial_eigen_method(A, inverse_iteration)
        else:
            print("\n(!) Некорректный ввод")


if __name__ == "__main__":
    main_menu()