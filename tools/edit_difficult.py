import os

with open("black_list.txt", "r") as f:
    lines = f.readlines()

black_list_dirs = []
for line in lines:
    possible_black_dir = line[:-6]
    if possible_black_dir not in black_list_dirs:
        black_list_dirs.append(possible_black_dir)

print(len(black_list_dirs))

with open("black_list_dirs.txt", "w") as file:
    for dir in black_list_dirs:
        file.write(dir + "\n")