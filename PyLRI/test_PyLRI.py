import PyLRI as lri
s=lri.Shape_Vector([1,2])
print(s)
t=lri.Tensor(s)
t.ptr()[0]=3
t.ptr()[1]=4
print(t(0,0))
print(t(0,1))

t1=lri.Tensor(s)
t1.ptr()[0]=5
t1.ptr()[1]=6
t2=t-t1
print(t2(0,0))
print(t2(0,1))
