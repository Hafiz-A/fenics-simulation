# File: simulation.py

# Import necessary libraries
from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# 1. Define simulation parameters
length = 0.025      # m
thickness = 0.005   # m
nx = 50             # Number of elements in x-direction
ny = 20             # Number of elements in y-direction

# 2. Create mesh and define function space for variables
mesh = RectangleMesh(Point(0, 0), Point(length, thickness), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# 3. Define physical and material properties
rho = 4430.0      # Density (kg/m^3)
Cp = 526.0        # Specific Heat (J/kg.K)
k = 6.7           # Thermal conductivity (W/m.K)
D0 = 1e-8         # Pre-exponential factor for diffusion (m^2/s)
Q = 100e3         # Activation energy for diffusion (J/mol)
R = 8.314         # Ideal gas constant (J/mol.K)

# 4. Time stepping parameters
dt = 60.0         # Time step (s)
T_end = 3600.0    # Total simulation time (s)
num_steps = int(T_end / dt)

# 5. Define initial conditions
# T_n holds the solution from the previous time step
T_n = interpolate(Constant(25.0), V)
C_n = interpolate(Constant(0.0), V)

# 6. Define boundary conditions on the bottom surface (y=0)
T_surface = 100.0  # Surface temperature in Celsius
C_surface = 5.0    # Surface concentration in Molar

class SurfaceBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

# Mark the boundary to apply the conditions
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
surface = SurfaceBoundary()
surface.mark(boundary_markers, 1) # Mark the surface with tag '1'

# Create Dirichlet boundary conditions
bc_T = DirichletBC(V, Constant(T_surface), boundary_markers, 1)
bc_C = DirichletBC(V, Constant(C_surface), boundary_markers, 1)

# 7. Set up the variational problems (the core of the FEM)
# Define trial and test functions
T = TrialFunction(V)
C = TrialFunction(V)
v = TestFunction(V)
w = TestFunction(V)

# Define the temperature-dependent diffusion coefficient as a Function
D = Function(V)

# Weak form for the Heat Equation (Implicit Euler)
F_T = (rho * Cp * T * v * dx) + (dt * k * dot(grad(T), grad(v)) * dx) - (rho * Cp * T_n * v * dx)
a_T, L_T = lhs(F_T), rhs(F_T)

# Weak form for the Diffusion Equation (Implicit Euler)
F_C = (C * w * dx) + (dt * D * dot(grad(C), grad(w)) * dx) - (C_n * w * dx)
a_C, L_C = lhs(F_C), rhs(F_C)

# 8. Prepare for time-stepping solution
T_solution = Function(V)
C_solution = Function(V)

print("Starting simulation...")
# 9. Time-stepping loop
for n in range(num_steps):
    current_time = (n + 1) * dt
    solve(a_T == L_T, T_solution, bc_T)

    D_expr = Expression('D0 * exp(-Q / (R * (T_k + 273.15)))',
                        degree=2, D0=D0, Q=Q, R=R, T_k=T_solution)
    D.assign(project(D_expr, V))

    solve(a_C == L_C, C_solution, bc_C)

    T_n.assign(T_solution)
    C_n.assign(C_solution)

    if (n + 1) % 10 == 0:
        print(f"Solved up to t = {current_time:.0f} s")

# 10. Plotting the final result
print("Simulation finished. Plotting final concentration.")

coords = mesh.coordinates()
triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], mesh.cells())
c_values = C_solution.vector().get_local()

plt.figure(figsize=(10, 5))
contour = plt.tricontourf(triangulation, c_values, levels=50, cmap='viridis')
plt.colorbar(contour, label='Sodium Concentration (Molar)')
plt.title(f'Final Sodium Concentration Profile at t = {T_end} s')
plt.xlabel('Length (m)')
plt.ylabel('Thickness (m)')
plt.gca().set_aspect('equal', adjustable='box')

# Use plt.show() to display the plot in the interactive notebook
plt.show()
