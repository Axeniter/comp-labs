from system_create import generate_random_system, create_input_system

def main_menu():
    print("МЕНЮ")
    print("="*35)
    print("1. Создать систему")
    print("2. Уйти")
    print("="*35)
    choice = input("\nВыбор: ")

    if choice == "1":
        create_system_menu()
    elif choice == "2":
        exit()
    else:
        print("Некорректный ввод\n")

def create_system_menu():
    while True:
        try:
            n = int(input("Введите размерность системы (n): "))
            break
        except ValueError:
            print("Неккоректный ввод\n")

    print("СОЗДАНИЕ СИСТЕМЫ")
    print("="*35)
    print("1. Случайная генерация")
    print("2. Ручной ввод")
    print("="*35)

    A, b = None, None
    while True:
        choice = input("\nВыбор: ")

        if choice == "1":
            while True:
                try:
                    d = int(input("Введите разброс: "))
                    break
                except ValueError:
                    print("Неккоректный ввод\n")

            A, b = generate_random_system(n, d)
            break
        elif choice == "2":
            A, b = create_input_system(n)
            break
        else:
            print("Некорректный ввод\n")
    
    select_method_menu(A, b)

def select_method_menu(A, b):
    if (A, b) == (None, None):
        return

def main():
    print("===== Вычислительная математика =====\n")
    while True:
        main_menu()

if __name__ == "__main__":
    main()