import numpy as np


def read_data_libsvm(path):
    m = int(input('number of features?'))  # features number!!!!!
    f1 = open(path, 'r')
    f2 = open("data_clean.txt", 'wt')
    for line in f1:
        cline = line[:]
        d = line.count(' ')
        for k in range(0, m-d):
            cline = cline + ' \' \''
        f2.write(cline)

    f1.close()
    f2.close()
    data_str = np.genfromtxt("data_clean.txt", delimiter=" ", dtype=str)
    data = []
    n = len(data_str)

    for i in range(0, n):
        data.append([])
        data[i].append(float(data_str[i][0]))
        for j in range(1, m+1):
            if str(j)+':' in str(data_str[i][j]):
                data[i].append(float(str(data_str[i][j]).replace(str(j)+':', '')))
            else:
                print()
                print(str(data_str[i][j]))
                data[i].append(np.nan)

    return data