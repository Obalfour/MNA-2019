import numpy as np

##### WHAT IF WE HAVE A 0 IN X[O] ?? ##### 

# 'a' must be square
a = np.array(
	[[12, -51, 4], 
    [6, 167, -68],
    [-4, 24, -41]
    ])
print(a)

# Information needed for the iteration (QR decomposition)
# m = gives me the amount of arrays
# n = gives me the amount of elements in each array 
m,n = a.shape
# turns the array into a matrix 
R = np.asmatrix(a)		# Transformed matrix so far
# gives a mxm matrix with 1's in the diagonal
Q = np.eye(m) 			# Orthogonal transform so far
# P = I - 2vv' ; Householder matrix
# v ; Householder vector, if v is not unit vector, we need to normalize it
# b = 2/||v||^2 ; so we can simply write P=I-bvv'

for i in range(n):

	normx = np.linalg.norm(R[i:, i])
	# grabs columns, from the diagonal down 
	x = R[i:, i]
	# we could choose any p, but often its p = -sign(xi) ; this is better for round off errors
	# sign returns: -1 if x < 0, 0 if x==0, 1 if x > 0
	p = np.multiply(-1,np.sign(x[0]))
	u = x[0] - np.multiply(normx, p)
	v = np.divide(x,u)
	#v = x/u
	v[0] = 1
	b = np.divide(np.multiply(np.multiply(-1,p),u), normx)

	R[i:, :] = R[i:, :] - np.multiply(b,np.outer(v, v).dot(R[i:, :]))
	Q[:, i:] = Q[:, i:] - np.multiply(b,Q[:, i:].dot(np.outer(v, v)))

print(Q)
print(R)