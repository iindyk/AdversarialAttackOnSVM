a = 1e9
b = 1e-6
for i in range(0, int(1e6)):
    a += b
print(a - 1e9)