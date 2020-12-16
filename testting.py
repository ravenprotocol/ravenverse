import ravop.core as R

a = R.Tensor([10, 100])
b = R.Tensor([10,12344])
f = R.div(b,a)
print(f.output)
