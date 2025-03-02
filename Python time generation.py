import numpy as np
import scipy.integrate
import torch
import math
from scipy.optimize import fsolve
from sympy import symbols, lambdify, I, latex
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, TensorDataset

# Define all the variables
x = symbols("x")
x_bar = symbols(r"\bar{x}")
omega_0 = symbols(r"\omega_0", real = True)

# Define Hamilton's equations
def h_2n(n_val, x_val, x_bar_val, omega_0_val):
	return ((-1)**(n_val+1) * omega_0_val**2 / math.factorial(2*n_val)) * (1 / (2*omega_0_val))**n_val * (x_val + x_bar_val)**(2*n_val)

# Define the Hamiltonian function: h 
h = omega_0 * x * x_bar

for i in range(2,4):
	h += h_2n(i, x, x_bar, omega_0)

# Differentiate with respect to x_bar
dhdx_bar = lambdify([x, x_bar, omega_0], h.diff(x_bar) * I)

def hamiltonian_system(omega):
    def diff_func(time, z):
        x_complex = z[0] + 1j * z[1]  
        dxdt = dhdx_bar(x_complex, x_complex.real - 1j * x_complex.imag, omega)
        return [dxdt.real, dxdt.imag]
    return diff_func

def solve_for_x(H, omega_0_val):
    equation = lambdify(x, h.subs({x_bar: x, omega_0: omega_0_val}) - H)
    x0_guess = np.sqrt(H / omega_0_val)
    x_solution = fsolve(equation, x0_guess)
    return x_solution[0]

def Hamiltonian_encoder_gen(init_H, init_omega, t_span, t_steps):
	t_eval = np.linspace(*t_span, t_steps)

	# Set initial condition for x and x̄ based on chosen 
	# energy H as sqrt(H/omega_0) with phase as 0
	x0 = solve_for_x(init_H, init_omega)
	# x0 = fsolve(lambda x: h_f(x + 0j, 0 + 0j, 1).real - H, [2])
	z0 = [x0.real, x0.imag]

	diff_func = hamiltonian_system(init_omega)

	# Solve the ODEs
	sol = solve_ivp(diff_func, t_span, z0, t_eval=t_eval, rtol=1e-8, atol=1e-8)
	x_sol = sol.y[0] + 1j * sol.y[1]

	# Compute action variable J and action angle phi
	J_sol = np.abs(x_sol) ** 2
	phi_sol = np.angle(x_sol)

	return x_sol, J_sol, phi_sol