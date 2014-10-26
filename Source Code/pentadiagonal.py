import numpy as np

import scipy.sparse as sparse

import scipy.stats as stats

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

S_0 = 100.0
K = 90.0
r = 0.0025
q = 0.0125
T = 1.0
sigma = 0.5
S_min = 0.0
S_max = 250.0
N = 1000
M = 250

dS = (S_max - S_min) / N
dtau = T / M

def S(j):
	return S_min + j * dS
def tau(k):
	return k * dtau

def alpha(j, k):
	return sigma * sigma * S(j) * S(j) * dtau / (24 * dS * dS)

def beta(j, k):
	return (r - q) * S(j) * dtau / (12 * dS)

def a(j, k):
	return -alpha(j, k) + beta(j, k)

def b(j, k):
	return 16 * alpha(j, k) - 8 * beta(j, k)

def c(j, k):
	return 1 - r * dtau - 30 * alpha(j, k)

def d(j, k):
	return 16 * alpha(j, k) + 8 * beta(j, k)

def e(j, k):
	return -alpha(j, k) - beta(j, k)

def aa(j, k):
	return -a(j, k)

def bb(j, k):
	return -b(j, k)

def cc(j, k):
	return 2 - c(j, k)

def dd(j, k):
	return -d(j, k)

def ee(j, k):
	return -e(j, k)

def A_EXP(k):
	A = map(lambda j: a(j, k) , range(4,N-1)) 
	B = map(lambda j: b(j, k) , range(3,N-1)) 
	C = map(lambda j: c(j, k) , range(2,N-1))
	D = map(lambda j: d(j, k) , range(2,N-2))
	E = map(lambda j: e(j, k) , range(2,N-3))
	data = [A, B, C, D, E]

	# third order Neumann boundary conditions
	f1 = np.array([-2.84210526315790, 1.68421052631579, 0.157894736842105])
	f2 = np.array([-2.05263157894737, 1.10526315789474, -0.0526315789473684])
	f3 = np.array([-16.0, 30.0, -16.0, 1.0])

	C[0] = C[0] - f1[0] * a(2, k) - f2[0] * b(2, k)
	D[0] = D[0] - f1[1] * a(2, k) - f2[1] * b(2, k)
	E[0] = E[0] - f1[2] * a(2, k) - f2[2] * b(2, k)

	B[0] = B[0] - f3[0] * a(3, k)
	C[1] = C[1] - f3[1] * a(3, k)
	D[1] = D[1] - f3[2] * a(3, k)
	E[1] = E[1] - f3[3] * a(3, k)

	C[-1] = C[-1] - f1[0] * e(N-2, k) - f2[0] * d(N-2, k)
	B[-1] = B[-1] - f1[1] * e(N-2, k) - f2[1] * d(N-2, k)
	A[-1] = A[-1] - f1[2] * e(N-2, k) - f2[2] * d(N-2, k)

	D[-1] = D[-1] - f3[0] * e(N-3, k)
	C[-2] = C[-2] - f3[1] * e(N-3, k)
	B[-2] = B[-2] - f3[2] * e(N-3, k)
	A[-2] = A[-2] - f3[3] * e(N-3, k)

	offsets = np.array([-2, -1, 0, 1, 2])
	return sparse.diags(data, offsets)

def A_IMP(k):
	A = map(lambda j: aa(j, k) , range(4,N-1)) 
	B = map(lambda j: bb(j, k) , range(3,N-1)) 
	C = map(lambda j: cc(j, k) , range(2,N-1))
	D = map(lambda j: dd(j, k) , range(2,N-2))
	E = map(lambda j: ee(j, k) , range(2,N-3))
	data = [A, B, C, D, E]

	# third order Neumann boundary conditions
	f1 = np.array([-2.84210526315790, 1.68421052631579, 0.157894736842105])
	f2 = np.array([-2.05263157894737, 1.10526315789474, -0.0526315789473684])
	f3 = np.array([-16.0, 30.0, -16.0, 1.0])

	C[0] = C[0] - f1[0] * aa(2, k) - f2[0] * bb(2, k)
	D[0] = D[0] - f1[1] * aa(2, k) - f2[1] * bb(2, k)
	E[0] = E[0] - f1[2] * aa(2, k) - f2[2] * bb(2, k)

	B[0] = B[0] - f3[0] * aa(3, k)
	C[1] = C[1] - f3[1] * aa(3, k)
	D[1] = D[1] - f3[2] * aa(3, k)
	E[1] = E[1] - f3[3] * aa(3, k)

	C[-1] = C[-1] - f1[0] * ee(N-2, k) - f2[0] * dd(N-2, k)
	B[-1] = B[-1] - f1[1] * ee(N-2, k) - f2[1] * dd(N-2, k)
	A[-1] = A[-1] - f1[2] * ee(N-2, k) - f2[2] * dd(N-2, k)

	D[-1] = D[-1] - f3[0] * ee(N-3, k)
	C[-2] = C[-2] - f3[1] * ee(N-3, k)
	B[-2] = B[-2] - f3[2] * ee(N-3, k)
	A[-2] = A[-2] - f3[3] * ee(N-3, k)

	offsets = np.array([-2, -1, 0, 1, 2])
	return sparse.diags(data, offsets)

def BS_PUT(j, k):
	if (j == 0): return 0
	F = S(j) * np.exp((r - q) * tau(k))
	d1 = (np.log(F / K) + 0.5 * sigma * sigma * tau(k)) / (sigma * np.sqrt(tau(k)))
	d2 = d1 - sigma * np.sqrt(tau(k))
	return np.exp(-r * tau(k)) * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))

V_mat = np.matrix(map(lambda j: max(K - S(j), 0), range(2,N-1))).H
V = list([np.hstack([np.array(V_mat).squeeze()])])
for k in range(1, M+1):
	RHS = A_EXP(k)+ sparse.eye(N-3, format='dia')
	LHS = A_IMP(k+1) + sparse.eye(N-3, format='dia')
	RHS_PROD = RHS * V_mat
	V_mat = sparse.linalg.spsolve(LHS, RHS_PROD)
	V_arr = np.array(V_mat).squeeze()
	V.append(np.hstack([V_arr]))

V_vec = V[M]
BS_vec = np.array(map(lambda j: BS_PUT(j, M), range(2,N-1)))
S_vec = np.array(map(lambda j: S(j), range(2,N-1)))
Tau_vec = np.array(map(lambda k: tau(k), range(0,M+1)))
DIFF_vec = BS_vec - V_vec

print "The numerical price of the put option with S_0 = {} is {}".format(S_0, V_vec[np.where(S_vec == S_0)].item())
print "The theoretical price of the put option with S_0 = {} is {}".format(S_0, BS_vec[np.where(S_vec == S_0)].item())

fig, ax = plt.subplots()
plt.title('Solution from Solver with Pentadiagonal Stiffness Matrix')
ax.set_xlabel('S_0')
ax.set_ylabel('Numerical Solution')
plt.plot(S_vec, V_vec)
plt.show()

fig, ax = plt.subplots()
plt.title('Accuracy of Solver with Pentadiagonal Stiffness Matrix')
ax.set_xlabel('S_0')
ax.set_ylabel('Theoretical Price Minus Numerical Solution')
plt.plot(S_vec, DIFF_vec)
plt.show()

fig, ax = plt.subplots()
plt.title('Solution from Solver with Pentadiagonal Stiffness Matrix')
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(S_vec, Tau_vec)
Z = np.array(V)
ax.plot_surface(X, Y, Z)
ax.set_xlabel('S_0')
ax.set_ylabel('Tau')
ax.set_zlabel('Numerical Solution')
plt.show()


