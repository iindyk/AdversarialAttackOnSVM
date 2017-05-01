import numpy as np


def read_data(path):
    m = int(input('number of features?'))  # features number!!!!!
    f1 = open(path, 'wt')
    f2 = open("data_clean.txt", 'w')
    for line in f1:
        cline = line[:]
        d = line.count(' ')
        for k in range(0, m-d):
            cline += ' \' \''
        f2.write(cline)

    f1.close()
    f2.close()
    data_str = np.genfromtxt("data_clean.txt", delimiter=" ", dtype=None)
    data = []
    n = data_str.size

    for i in range(0, n):
        data.append([])
        data[i].append(float(data_str[i][0]))
        for j in range(1, m):
            if 'j'+':' in data_str[i][j]:
                data[i].append(float(data_str[i][j].replace(str(j)+':', '')))
            else:
                data[i].append(np.nan)

    return data