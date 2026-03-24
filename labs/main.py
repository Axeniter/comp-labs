from system_create import generate_random_system, create_input_system
from gauss_method import gauss_method
from optimal_exclusion import optimal_exclusion
from utils import check_singular, print_system, execute_method

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
        print("2. Ручной ввод")
        print("3. Назад")
        print("="*50)
        
        choice = input("\nВыбор: ").strip()
        
        if choice == "1":
            random_system_flow()
        elif choice == "2":
            input_system_flow()
        elif choice == "3":
            return
        else:
            print("\n(!) Некорректный ввод")

def random_system_flow():
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
            d = int(input("Введите разброс (диапазон [-d, d]): "))
            if d <= 0:
                print("(!) Разброс должен быть положительным")
                continue
            break
        except ValueError:
            print("(!) Введите целое положительное число")
    
    A, b = generate_random_system(n, d)
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
        print("1. Метод Гаусса с поиском максимального по матрице")
        print("2. Метод оптимального исключения")
        print("3. Назад")
        print("="*50)
        
        choice = input("\nВыбор: ").strip()
        
        if choice == "1":
            if check_singular(A):
                continue
            execute_method(A, b, gauss_method)    
        elif choice == "2":
            if check_singular(A):
                continue
            execute_method(A, b, optimal_exclusion)
        elif choice == "3":
            return
        else:
            print("\n(!) Некорректный ввод")

if __name__ == "__main__":
    main_menu()