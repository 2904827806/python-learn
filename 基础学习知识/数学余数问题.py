a = 0
for i in range(10000000000):
    a +=1
    b = a%3
    c = a%5
    d = a%7
    if int(b) == 2 and (int(c) == 3) and (int(d) == 2):

        print(a)
        break
    else:
        continue
    
