 # Read in the file
f1 = open('data.txt', 'r')
f2 = open('data_clean.txt', 'w')
for line in f1:
    f2.write(line.replace(' 1:', ' ').replace(' 2:', ' '))
f1.close()
f2.close()