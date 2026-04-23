from system_create import generate_random_system, create_input_system
from gauss_method import gauss_method
from optimal_exclusion import optimal_exclusion
from cholesky_method import cholesky_method
from lu_decomposition import lu_decomposition
from bordering_method import bordering_method
from qr_decomposition import qr_decomposition
from jacobi_method import jacobi_method, check_jacobi_convergence
from utils import is_singular, print_system, execute_method, is_symmetrical


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
        print("3. Случайная генерация (симметричная, положительно определённая матрица)")
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
            random_system_flow(symmetrical=True, positive_definite=True)
        elif choice == "4":
            random_system_flow(diagonally_dominant=True)
        elif choice == "5":
            input_system_flow()
        elif choice == "6":
            return
        else:
            print("\n(!) Некорректный ввод")


def random_system_flow(symmetrical=False, positive_definite=False, diagonally_dominant=False):
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
    
    A, b = generate_random_system(n, d,
                                  symmetrical=symmetrical,
                                  positive_definite=positive_definite,
                                  diagonally_dominant=diagonally_dominant)
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
        print("2. Метод Гаусса с поиском максимального по матрице")
        print("3. Метод оптимального исключения")
        print("4. Метод квадратного корня (метод Холецкого)")
        print("5. Метод LU-разложения")
        print("6. Метод окаймления")
        print("7. Метод отражений (QR-разложение)")
        print("8. Метод Якоби")
        print("9. Метод Релаксации")
        print("10. Метод сопряженных градиентов")
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
                print("Метод Якоби не сходится по критерию сходимости")
                continue
            execute_method(A, b, jacobi_method)
        elif choice == "9":
            continue
        elif choice == "10":
            continue
        else:
            print("\n(!) Некорректный ввод")


if __name__ == "__main__":
    main_menu()