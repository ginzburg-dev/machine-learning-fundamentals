import numpy as np    # it is an unofficial standard to use np for numpy
import time

# Vector creation

# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4);    print(f"np.zeros(4) :    a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));    print(f"np.zeros((4,)) :    a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4);    print(f"np.random.random_sample(4) :    a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4);   print(f"np.arange(4) :    a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);    print(f"np.random.rand(4) :    a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill with user specified values
a = np.array([5, 4, 3, 2]);     print(f"np.array([5, 4, 3, 2]) :    a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5., 4, 3, 2]);     print(f"np.array([5, 4, 3, 2]) :    a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# Indexing

#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)
#access an element
print(f"a[2].shape: {a[2].shape} a[2]= {a[2]}, Accessing an element returns a scalar")
#access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}, a[-2] = {a[-2]}")
#indexes must be written within the range of the vector or they will produce an error
try:
    c = a[10]
except Exception as e:
    print("Caught an exception: ")
    print(e)

# Slicing

a = np.arange(10)
print(f"a = {a}")
#access 5 consecutive elements (start:stop:step)
c = a[2:7:1];   print(f"a[2:7:1] = {c}") # prints [2, 3, 4, 5, 6]. start - inclusive, stop - exclusive
#access 3 elements separated by two
c = a[2:7:2];   print(f"a[2:7:2] = {c}")
#access all elements index 3 and above
c = a[3:];   print(f"a[3:] = {c}")
#access all elements below index 3
c = a[:3];   print(f"a[:3] = {c}")
#access all elements
c = a[:];   print(f"a[:] = {c}")

# Single vector operations

a = np.array([1, 2, 3, 4, 5])
print(f"a = {a}")
#negate elaments of a
b = -a;     print(f"b = -a  : {b}")
#sum all elements of a and return scalar
b = np.sum(a);     print(f"b = np.sum(a)  : {b}")
#mean of a
b = np.mean(a);     print(f"b = np.mean(a)  : {b}")
#sqr of all elements
b = a**2;     print(f"b = a**2  : {b}")

# Vector element-wise operations

#add two vectors
a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
print(f"a + b = {a + b}")

#try a mismatched vector operation
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print(f"Error: {e}")

# Scalar vector operations

a = np.array([1, 2, 3, 4])
b = 5 * a
print(f"b = 5 * a : {b}")

# Vector dot product

def my_dot(a, b):
    x=0
    for i in range(a.shape[0]):
        x += a[i] * b[i]
    return x

a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
print(f"my_dot(a, b) = {my_dot(a, b)}")

#np.dot
a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
c = np.dot(a, b)
print(f"np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape}")
c = np.dot(b, a)
print(f"np.dot(b, a) = {c}, np.dot(b, a).shape = {c.shape}")

# Speed test

np.random.seed(1)
a = np.random.rand(10000000)
b = np.random.rand(10000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(f"np.dot(a, b) = {c:.4f}")
print(f"Vectorized version duration: {1000*(toc - tic):.4f} ms")

tic = time.time()
c = my_dot(a, b)
toc = time.time()

print(f"my_dot(a, b) = {c:.4f}")
print(f"Loop version duration: {1000*(toc - tic):.4f} ms")

del(a); del(b) #remove this big arrays from memory

#common example
X = np.array([[1], [2], [3], [4]])
w = np.array([2])
c = np.dot(X[1], w)
print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

# Matrices

a = np.zeros((1, 5))
print(f"a shape = {a.shape}, a = {a}")

a = np.zeros((2, 1))
print(f"a shape = {a.shape}, a = {a}")

a = np.random.random_sample((1, 1))
print(f"a shape = {a.shape}, a = {a}")

a = np.array([[5], [4], [3]]); print(f"a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],
            [4],
            [3]])
print(f"a shape = {a.shape}, np.array: a = {a}") # the same

# Matrix indexing

a = np.arange(6).reshape(-1, 2); print(f"np.arange(6).reshape(-1, 2) : {a}") # [[0 1] [2 3] [4 5]]
b = np.arange(6); print(f"np.arange(6) : {b}") # [0 1 2 3 4 5]

print(f"\na[2, 0].shape: {a[2, 0].shape}, a[2, 0] = {a[2, 0]}, type(a[2, 0]) = {type(a[2, 0])}") # a[2, 0].shape: (), a[2, 0] = 4, type(a[2, 0]) = <class 'numpy.int64'>

print(f"\na[2].shape: {a[2].shape}, a[2] = {a[2]}, type(a[2]) = {type(a[2])}") 

#Slicing

a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

print(f"a[0, 2:7:1] = {a[0, 2:7:1]}, a[0, 2:7:1].shape = {a[0, 2:7:1].shape}, a 1-D array")

print(f"a[:, 2:7:1] = {a[:, 2:7:1]}, a[:, 2:7:1].shape = {a[:, 2:7:1].shape}, a 2-D array")

print(f"a[:,:] = {a[:,:]}, a[:,:].shape = {a[:,:].shape}, a 2-D array")

print(f"a[1,:] = {a[1,:]}, a[1,:].shape = {a[1,:].shape}, a 1-D array")
print(f"a[1] = {a[1]}, a[1].shape = {a[1].shape}, a 1-D array") # the same
