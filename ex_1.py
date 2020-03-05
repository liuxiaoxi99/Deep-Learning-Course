import math
a=int(input("输入a的值："))
b=int(input("输入b的值："))
c=int(input("输入c的值："))
d=b**2-4*a*c
if d>0:
    x1=(-b+math.sqrt(d))/(2*a)
    x2=(-b-math.sqrt(d))/(2*a)
    print("方程有两个解：x1=%f,x2=%f"%(x1,x2))
elif d==0:
    x=-b/2*a
    print("方程有一个解：x=%f"%(x))
else:
    print("方程无解")