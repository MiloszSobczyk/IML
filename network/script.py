import os
import random
import shutil

# Ścieżki do folderów
base_dir = "./data"
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# Tworzenie folderów test/accept i test/refuse
os.makedirs(os.path.join(test_dir, "accept"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "refuse"), exist_ok=True)


# Funkcja do przenoszenia plików
def move_half_to_test(category):
    source_dir = os.path.join(validation_dir, category)
    dest_dir = os.path.join(test_dir, category)

    # Pobieranie listy plików w folderze
    files = os.listdir(source_dir)

    # Losowe wybieranie połowy plików
    random.shuffle(files)
    num_to_move = len(files) // 2
    files_to_move = files[:num_to_move]

    # Przenoszenie wybranych plików
    for file in files_to_move:
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.move(src_path, dest_path)
        print(f"Moved: {src_path} -> {dest_path}")


# Przenoszenie danych dla kategorii "accept" i "refuse"
move_half_to_test("accept")
move_half_to_test("refuse")

print("Finished moving files!")
