import time

import ravop.core as R
from ravcom import inform_server

a = R.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
b = R.Scalar(10)

c = R.add(a, b)
d = R.sub(a, b)
e = R.multiply(a, b)
f = R.mean(a)
g = R.median(a)

inform_server()

# Wait for 10 seconds
time.sleep(10)

print(c())
print(d())
print(e())
print(f())
print(g())
