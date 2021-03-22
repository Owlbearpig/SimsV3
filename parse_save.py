save_file = 'save_2.txt'
with open(save_file) as file:
    file_content = file.read()
    splits = file_content.split(' ')
    best_min = 1
    for split in splits:
        if '0.000' in split and abs(float(split)) < best_min:
            best_min = float(split)

print(best_min)