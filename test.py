import ravop.core as R

a = R.Scalar(10)
b = R.Scalar(12)

c = a.add(b)

d = a.greater(b)

e = a.less(b)

print(c.output, d.output, e.output)
