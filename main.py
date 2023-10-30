import numpy

# ===============================================================================================

# class Matrix:
# 	data = []
# 	m = 0
# 	n = 0

# 	def transpose(self):
# 		out = []
# 		for i in range(self.n):
# 			out.append([xs[i] for xs in self.data])
# 		return Matrix(self.n, self.m, out)

# 	def column_vec(self, i):
# 		if i >= self.n:
# 			raise Exception(f"Column vector {i} does not exist in a {self.m}x{self.n} Matrix")
# 		return Matrix(1, self.m, [[xs[i] for xs in self.data]]).transpose()
	
# 	def row_vec(self, i):
# 		if i >= self.m:
# 			raise Exception(f"Row vector {i} does not exist in a {self.m}x{self.n} Matrix")
# 		return Matrix(1, self.n, [self.data[i]]).transpose()

# 	def __repr__(self):
# 		out = ""
# 		for xs in self.data:
# 			out += str(xs) + "\n"
# 		return out[:-1]

# 	def __init__(self, m, n, data = None):
# 		if data == None:
# 			data = [[0 for j in range(n)] for i in range(m)]
# 		elif len(data) != m or sum(map(lambda xs: len(xs), data)) != n*m:
# 			raise Exception(f"Invalid data input for {m}x{n} Matrix: {data}")
		
# 		self.data = data
# 		self.m = m
# 		self.n = n

# 	def mul(self, m):
# 		out = []
# 		for c in range(m.n):
# 			vec = Matrix(self.m, 1)
# 			for i in range(self.n):
# 					e = m.column_vec(c).data[i][0]
# 					vec += e*self.column_vec(i)
# 			out.append([xs[0] for xs in vec.data])
# 		return Matrix(m.n, self.m, out).transpose()
	
# 	def dot(self, m):
# 		return self.transpose().mul(m)

# 	def __mul__(self, x):
# 		out = [[x * y for y in ys] for ys in self.data]
# 		return Matrix(self.m, self.n, out)

# 	__rmul__ = __mul__

# 	def __add__(self, m):
# 		if m.m != self.m or m.n != self.n:
# 			raise Exception(f"Cannot add matrices of size {self.m}x{self.n} and {m.m}x{m.n}")
# 		return Matrix(self.m, self.n, [[self.data[r][c] + m.data[r][c] for c in range(self.n)] for r in range(self.m)])

# ===============================================================================================

def linear_regression(points):
	# Construct system of equations from y = mx + c
	# [1, x_1] [c] = [y_1]
	# [1, ...] [m]	 [...]
	# [1, x_k]		 [y_k]
	A = numpy.array([[1, point[0]] for point in points])
	At = A.transpose()
	b = numpy.array([[point[1]] for point in points])

	# Find the least square solution to the system
	# At.A.x = At.b
	# x = (At.A)^-1.At.b
	x = numpy.linalg.inv(At.dot(A)).dot(At).dot(b)
	return x

def linear_regression_qr(points):
	# Construct system of equations from y = mx + c
	# [1, x_1] [c] = [y_1]
	# [1, ...] [m]	 [...]
	# [1, x_k]		 [y_k]
	A = numpy.array([[1, point[0]] for point in points])
	Q, R = numpy.linalg.qr(A)
	b = numpy.array([[point[1]] for point in points])

	# Find the least square solution to the system
	# At.A.x = At.b
	# (Q.R)t.(Q.R).x = (Q.R)t.b
	# Rt.Qt.Q.R.x = Rt.Qt.b
	# Rt.R.x = Rt.Qt.b
	# R.x = Qt.b
	Rx = Q.transpose().dot(b)
	xy = Rx[1] / R[1, 1]
	xx = (Rx[0] - xy * R[0, 1]) / R[0, 0]
	x = numpy.array([xx, xy])
	return x

if __name__ == "__main__":
	print(linear_regression_qr([[1, 2], [3, 4]]))

