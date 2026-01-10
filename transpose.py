def transpose(m):
    rows = len(m)
    col = len(m[0])
    t = [[0] * rows for i in range(col)]
    for i in range(rows):
        for j in range(col):
            t[j][i] = m[i][j]

    return t

rows=int(input("enter no of rows: "))
col = int(input("enter no of cols: "))

m=[]
print("enter elements:")
for i in range(rows):
    m.append(list(map(int, input().split())))

result = transpose(m)

print("transpose = ")
for row in result:
    print(*row)
