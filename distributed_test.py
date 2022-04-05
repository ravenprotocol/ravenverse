import ravop as R
import time

R.initialize("distributed_test")
algo = R.Graph(name='lin_reg', algorithm='linear_regression', approach='distributed')

a = R.t([1, 2, 3]*10)
b = R.t([5, 22, 7]*10)
c = a + b

e = R.sum(c)
d = c + e + a
print('c: ', c())

p = R.argmax(d)
print('p: ', p())

w = R.t([1, 2, 3, 3, 4, 5])
f = R.max(d)
print('f: ', f())
x = R.sort(d)
print('x: ', x())
algo.end()