str1 = '移舟泊烟渚'
str2 ='日暮客愁新'
str3 ='野旷天低树'
str4 ='江清月近人'
v = [list(str1),list(str2),list(str3),list(str4)]
print(''*5,"横版\n")
for i in range (4):
    for j in range(5):
        if j ==4:
            print(v[i][j])
        else:
            print(v[i][j],end='')
print()
v.reverse()
print(v)
print(''*5,"竖版\n")
for i in range (5):
    for j in range(4):
        if j ==3:
            print(v[j][i])
        else:
            print(v[j][i],end='')
