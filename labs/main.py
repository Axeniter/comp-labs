from system_create import generate_random_system, create_input_system, generate_symmetrical_system
from gauss_method import gauss_method
from optimal_exclusion import optimal_exclusion
from cholesky_method import cholesky_method
from lu_decomposition import lu_decomposition
from bordering_method import bordering_method
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
        print("3. Ручной ввод")
        print("4. Назад")
        print("="*50)
        
        choice = input("\nВыбор: ").strip()
        
        if choice == "1":
            random_system_flow()
        elif choice == "2":
            random_system_flow(symmetrical=True)
        elif choice == "3":
            input_system_flow()
        elif choice == "4":
            return
        else:
            print("\n(!) Некорректный ввод")


def random_system_flow(symmetrical=False):
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
    
    if not symmetrical:
        A, b = generate_random_system(n, d)
    else:
        A, b = generate_symmetrical_system(n, d)
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
                print("(!) Матрица несимметрична")
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
        else:
            print("\n(!) Некорректный ввод")


if __name__ == "__main__":
    main_menu()