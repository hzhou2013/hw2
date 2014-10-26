Code Execution

A script to solve the Black-Scholes PDE for a put option with the specified parameters in the problem is located at pentadiagonal.py (4th order in S and 2nd order in tau) and tridiagonal.py (2nd order in both S and tau). Second order accuracy in tau is achieved by extending the Crank-Nicholson scheme.

The python scripts should run without modifications, provided that the following packages are available: numpy, scipy, matplotlib, matplotlib, and mpl_toolkits.


Boundary Conditions

1. Tridiagonal Model

The boundary condition is a second order Neumann condition (gamma = 0)

V_0 - 2 V_1 + V_2 = 0 -> V_0 = 2 V_1 - V_2    (2nd order central difference)

The condition at the upper boundary is analogous

V_N - 2 V_{N-1} + V_{N-2} = 0 -> V_N = 2 V_{N-1} - V_{N-2}

The expressions above allows us to get write the updates of V_1 to V_{N-1} as a function of the values of V_1 to V_{N-1} in the previous iteration.

2. Pentadiagonal Model

The boundary condition is a third order Neumann condition (gamma = 0)

35/12 V_0 - 26/3 V_1 + 19/2 V_2 - 14/3 V_3 + 11/12 V_4 = 0  (3rd order forward difference)

−1/12 V_0 + 4/3 V_1 - 5/2 V_2 - 4/3 V_3	- 1/12 V_4 = 0 (4th order central difference)

Using row reduction, we see that the equations above implies

V_0 = 2.84210526315790 V_2 - 1.68421052631579 V_3 - 0.157894736842105 + V_4
V_1 = 2.05263157894737 V_2 - 1.10526315789474 V_3 + 0.052631578947368 + V_4

The condition at the upper boundary is analogous:

V_N = 2.84210526315790 V_{N-2} - 1.68421052631579 V_{N-3} - 0.157894736842105 + V_{N-4}
V_{N-1} = 2.05263157894737 V_{N-2} - 1.10526315789474 V_{N-3} + 0.052631578947368 + V_{N-4}

The expressions above allows us to get write the updates of V_2 to V_{N-2} as a function of the values of V_2 to V_{N-2} in the previous iteration.
