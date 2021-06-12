import os
import pathlib
from pathlib import Path

name_filter = ['eigenstate_azimuths']

# get all csv files
csv_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(os.getcwd())
             for name in files
             if name.endswith('.csv')]

# filter csv files
csv_files2change = []
for csv_file in csv_files:
    for result in name_filter:
        if result in str(csv_file) and 'jump' not in str(csv_file):
            csv_files2change.append(csv_file)

for csv_file in csv_files2change:
    with open(csv_file) as file:
        lines = file.readlines()
        line_cnt = len(lines)

        # output file
        new_csv_path = Path(csv_file).parent / (Path(csv_file).stem + '_Wjump.csv')
        new_csv_file = open(new_csv_path, 'w')

        # Skip first column (#, ...)
        header = ''
        for column_name in lines[0].split(',')[1:]:
            header += column_name + ','
        header = header[:-1]  # remove last comma
        new_csv_file.write(header)

        # assuming header and to prevent out of bounds when comparing
        for line_number in range(1, line_cnt-2):
            line1_content = lines[line_number].split(',')
            line2_content = lines[line_number+1].split(',')
            if len(line1_content) != len(line2_content):
                continue

            # write every original line again
            new_csv_file.write(','.join(line1_content[1:]))

            # go through all parameters and compare to find jumps.
            jump = False # only write line if discontinuity
            #new_line = f'{line1_content[1]},{line1_content[2]},{line1_content[3]},' # skip these
            new_line = f'{line1_content[1]},'  # skip these
            for p1, p2 in zip(line1_content[2:], line2_content[2:]):
                p1, p2 = float(p1), float(p2)
                if abs(p1-p2) > 50:
                    new_line += 'nan' + ','
                    jump = True
                else:
                    new_line += str(p1) + ','

            new_line = new_line[:-1]  # remove last comma
            new_line += '\n'  # os.linesep
            if jump:
                new_csv_file.write(new_line)
                jump = False
            print(new_line)
