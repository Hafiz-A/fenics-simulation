# FEniCS Phase-Field Simulation of Hydrothermal Treatment of Ti-6Al-4V
#
# This simulation models the conversion of a native oxide layer (phi=+1)
# into a porous titanate hydrogel (phi=-1) driven by the diffusion
# of an active chemical species (C, e.g., NaOH).
#
# Governing Equations:
# 1. Allen-Cahn for phase transformation (oxide -> gel):
#    d(phi)/dt = -L * [ (phi^3 - phi) - K_chem*C*phi - kappa*nabla^2(phi) ]
#
# 2. Diffusion of chemical 'C' with phase-dependent mobility:
#    d(C)/dt = nabla . ( M(phi) * nabla(C) )

from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Parameters (Physical and Numerical) ---

# Time control
T = 50.0          # Total simulation time
num_steps = 250   # Number of time steps
dt = T / num_steps  # Time step size

# Domain size
WIDTH = 2.0       # Width of the simulation domain
HEIGHT = 2.0      # Height of the simulation domain

# Mesh resolution
NX = 100
NY = 100

# Phase-field parameters for Allen-Cahn
L = Constant(1.0)       # Kinetic parameter (interface mobility)
epsilon = 0.04          # Interface thickness parameter (related to kappa)
kappa = Constant(epsilon**2) # Gradient energy coefficient (kappa = epsilon^2)
K_chem = Constant(2.0)    # Chemical reaction strength (coupling C to phi)

# Diffusion parameters
M_gel = Constant(1.0)     # High diffusivity in the gel/liquid phase (phi=-1)
M_oxide = Constant(1e-4)  # Low diffusivity in the dense oxide phase (phi=+1)

# Initial conditions
h_oxide = 0.4             # Initial height of the oxide layer from the bottom
C0 = 1.0                  # Initial concentration of NaOH in the liquid

# --- 2. Mesh and Function Space ---

# Create a 2D rectangular mesh
mesh = RectangleMesh(Point(0, 0), Point(WIDTH, HEIGHT), NX, NY)

# Define the mixed function space for (phi, C)
# P1 = Finite element of degree 1
P1 = FiniteElement('P', triangle, 1)
W = FunctionSpace(mesh, P1 * P1)

# --- 3. Initial Conditions ---

# Use a UserExpression to define the complex initial state
class InitialConditions(UserExpression):
    def eval(self, values, x):
        # Initial condition for phi (phase field)
        if x[1] <= h_oxide:
            values[0] = 1.0  # Oxide layer
        else:
            values[0] = -1.0 # Liquid/Gel region

        # Initial condition for C (concentration)
        if x[1] > h_oxide + epsilon: # Add buffer to avoid sharp initial gradient at interface
            values[1] = C0   # NaOH in the liquid
        else:
            values[1] = 0.0  # No NaOH inside the oxide

    def value_shape(self):
        return (2,)

# Create functions for current (w) and previous (w_n) time steps
w = Function(W)
w_n = Function(W)

# Interpolate the initial conditions into the 'previous' state function
w_n.interpolate(InitialConditions())
# Split the functions to access phi and C individually
phi_n, C_n = w_n.split()

# --- 4. Boundary Conditions ---

# Define a Dirichlet BC for the concentration at the top of the domain
# This represents the bulk solution with constant NaOH concentration
c_top = Constant(C0)
bc_c = DirichletBC(W.sub(1), c_top, 'on_boundary && x[1] > HEIGHT - DOLFIN_EPS')
bcs = [bc_c]

# --- 5. Variational Formulation (The Core PDEs) ---

# Get trial and test functions for the mixed system
p, q = TestFunctions(W)
# Split the solution 'w' to get phi and C for the current time step
phi, C = split(w)

# Define the phase-dependent diffusivity M(phi)
# Smoothly interpolates between M_oxide and M_gel based on phi_n
M = M_oxide + (M_gel - M_oxide) * (1.0 - phi_n) / 2.0

# Define the double-well potential for the phase field
# Derivative: f_phi = phi^3 - phi
f_phi = pow(phi, 3) - phi

# Chemical driving force term
# This term 'tilts' the double-well potential, making phi=-1 more favorable
# when C is high. We use phi instead of (1-phi^2) for stability.
f_chem = K_chem * C * phi

# Define the weak form of the Allen-Cahn equation (temporal term first)
F_phi = (phi - phi_n) / dt * p * dx \
      + L * dot(kappa * grad(phi), grad(p)) * dx \
      + L * (f_phi - f_chem) * p * dx

# Define the weak form of the Diffusion equation (temporal term first)
# We use phi_n for M(phi) to keep the system linear with respect to C
F_C = (C - C_n) / dt * q * dx \
    + dot(M * grad(C), grad(q)) * dx

# Combine the two weak forms into a single system
F = F_phi + F_C

# --- 6. Time-Stepping Loop ---

# Create VTK files for saving the output for visualization in ParaView
vtkfile_phi = File('output/phi.pvd')
vtkfile_c = File('output/C.pvd')

t = 0
for i in range(num_steps):
    # Update current time
    t += dt
    print(f'Step {i+1}/{num_steps}, Time: {t:.4f}')

    # Solve the nonlinear variational problem for the current time step
    solve(F == 0, w, bcs, solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})

    # Update the solution for the previous time step
    w_n.assign(w)

    # Save the results to file
    phi_out, c_out = w.split()
    phi_out.rename("phi", "phase_field")
    c_out.rename("C", "concentration")
    vtkfile_phi << (phi_out, t)
    vtkfile_c << (c_out, t)

# --- 7. Plotting the Final Result ---
print("Simulation finished. Plotting final state.")

plt.figure(figsize=(12, 5))

# Plot the final phase field
plt.subplot(1, 2, 1)
final_phi, _ = w.split()
plot(final_phi, title="Final Phase Field (phi)")
plt.colorbar(plot(final_phi), label="phi (+1: Oxide, -1: Gel)")

# Plot the final concentration field
plt.subplot(1, 2, 2)
_, final_c = w.split()
plot(final_c, title="Final Concentration Field (C)")
plt.colorbar(plot(final_c), label="Concentration")

plt.tight_layout()
plt.savefig("final_state.png")
plt.show()
