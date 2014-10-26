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
	return sigma * sigma * S(j) * S(j) * dtau / (2 * dS * dS)

def beta(j, k):
	return (r - q) * S(j) * dtau / (2 * dS)

def l(j, k):
	return alpha(j, k) - beta(j, k)

def d(j, k):
	return 1 - r * dtau - 2 * alpha(j, k)

def u(j, k):
	return alpha(j, k) + beta(j, k)

def ll(j, k):
	return -l(j, k)

def dd(j, k):
	return 2 - d(j, k)

def uu(j, k):
	return -u(j, k)

def A_EXP(k):
	D = map(lambda j: d(j, k) , range(1,N)) 
	L = map(lambda j: l(j, k) , range(2,N)) 
	U = map(lambda j: u(j, k) , range(1,N-1))
	D[0] = D[0] + 2 * l(1, k)
	U[0] = U[0] - l(1, k)
	D[N-2] = D[N-2] + 2 * u(N-1, k) 
	L[N-3] = L[N-3] - u(N-1, k)
	data = [D, L, U]
	offsets = np.array([0,-1,1])
	return sparse.diags(data, offsets)

def A_IMP(k):
	D = map(lambda j: dd(j, k) , range(1,N)) 
	L = map(lambda j: ll(j, k) , range(2,N)) 
	U = map(lambda j: uu(j, k) , range(1,N-1))
	D[0] = D[0] + 2 * ll(1, k)
	U[0] = U[0] - ll(1, k)
	D[N-2] = D[N-2] + 2 * uu(N-1, k) 
	L[N-3] = L[N-3] - uu(N-1, k)
	data = [D, L, U]
	offsets = np.array([0,-1,1])
	return sparse.diags(data, offsets)

def BS_PUT(j, k):
	if (j == 0): return 0
	F = S(j) * np.exp((r - q) * tau(k))
	d1 = (np.log(F / K) + 0.5 * sigma * sigma * tau(k)) / (sigma * np.sqrt(tau(k)))
	d2 = d1 - sigma * np.sqrt(tau(k))
	return np.exp(-r * tau(k)) * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))

V_mat = np.matrix(map(lambda j: max(K - S(j), 0), range(1,N))).H
V = list([np.hstack([np.array(V_mat).squeeze()])])
for k in range(1, M+1):
	RHS = A_EXP(k)+ sparse.eye(N-1, format='dia')
	LHS = A_IMP(k+1) + sparse.eye(N-1, format='dia')
	V_mat = sparse.linalg.spsolve(LHS, RHS * V_mat)
	V_arr = np.array(V_mat).squeeze()
	V.append(np.hstack([V_arr]))

V_vec = V[M]
BS_vec = np.array(map(lambda j: BS_PUT(j, M), range(1,N)))
S_vec = np.array(map(lambda j: S(j), range(1,N)))
Tau_vec = np.array(map(lambda k: tau(k), range(0,M+1)))
DIFF_vec = BS_vec - V_vec

print "The numerical price of the put option with S_0 = {} is {}".format(S_0, V_vec[np.where(S_vec == S_0)].item())
print "The theoretical price of the put option with S_0 = {} is {}".format(S_0, BS_vec[np.where(S_vec == S_0)].item())

fig, ax = plt.subplots()
plt.title('Solution from Solver with Tridiagonal Stiffness Matrix')
ax.set_xlabel('S_0')
ax.set_ylabel('Numerical Solution')
plt.plot(S_vec, V_vec)
plt.show()

fig, ax = plt.subplots()
plt.title('Accuracy of Solver with Tridiagonal Stiffness Matrix')
ax.set_xlabel('S_0')
ax.set_ylabel('Theoretical Price Minus Numerical Solution')
plt.plot(S_vec, DIFF_vec)
plt.show()

fig, ax = plt.subplots()
plt.title('Solution from Solver with Tridiagonal Stiffness Matrix')
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(S_vec, Tau_vec)
Z = np.array(V)
ax.plot_surface(X, Y, Z)
ax.set_xlabel('S_0')
ax.set_ylabel('Tau')
ax.set_zlabel('Numerical Solution')
plt.show()

