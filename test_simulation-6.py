# FEniCS 2D FEM - SHORT TEST VERSION
# This script is modified to run quickly for testing purposes.

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import matplotlib.animation as animation

# --- TEST VERSION CHANGES ---
# The following parameters have been reduced for a quick run.
# Original values are commented out for reference.

length = 0.025
thickness = 0.005
nx = 30  # Reduced mesh resolution (Original: 150)
ny = 10  # Reduced mesh resolution (Original: 150)

# Time stepping parameters
dt = 1.0
T_end = 60.0 # Reduced simulation time (Original: 3600)
num_steps = int(T_end / dt)

# Parametric ranges for a quick test (2 runs total)
surface_temperatures = [80.0, 100.0]  # Reduced set (Original: [60.0, 80.0, 100.0, 120.0])
NaOH_concentrations = [5.0]           # Reduced set (Original: [1.0, 3.0, 5.0, 7.0, 10.0])

# --- END OF TEST VERSION CHANGES ---


# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(length, thickness), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define thermal and diffusion properties
rho = 4430  # kg/m^3
Cp = 526  # J/kg.K
k = 6.7  # W/m.K
D0 = 1e-8  # m^2/s
Q = 100e3  # J/mol
R = 8.314  # J/mol.K

# Define boundary markers
class SurfaceBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)
surface = SurfaceBoundary()
surface.mark(boundary_markers, 1)

# Create results directory
if not os.path.exists("Results"):
    os.makedirs("Results")
if not os.path.exists("CSV_Results"):
    os.makedirs("CSV_Results")
if not os.path.exists("Animations"):
    os.makedirs("Animations")
if not os.path.exists("Temperature_Profiles"):
    os.makedirs("Temperature_Profiles")

# Define sodium concentration threshold for layer thickness calculation
concentration_threshold = 0.1

# Prepare summary report
summary_data = []

# Start parametric study
for T_surface in surface_temperatures:
    for C_surface in NaOH_concentrations:

        print(f"--- Running TEST simulation for T_surface={T_surface}°C, NaOH={C_surface}M ---")

        # Define initial conditions
        T_init = Constant(25.0)
        C_init = Constant(0.0)

        T = interpolate(T_init, V)
        C = interpolate(C_init, V)

        bc_T = DirichletBC(V, Constant(T_surface), boundary_markers, 1)
        bc_C = DirichletBC(V, Constant(C_surface), boundary_markers, 1)

        # Define variational problem for heat equation
        Tn = Function(V)
        Tn.assign(T)
        T_trial = TrialFunction(V)
        v = TestFunction(V)
        F_T = rho * Cp * (T_trial - Tn) / dt * v * dx + k * dot(grad(T_trial), grad(v)) * dx
        T_problem, T_rhs = lhs(F_T), rhs(F_T)

        # Define variational problem for diffusion equation
        Cn = Function(V)
        Cn.assign(C)
        C_trial = TrialFunction(V)
        w = TestFunction(V)

        # Prepare output arrays
        times = []
        T_profiles = []
        C_profiles = []

        # Time-stepping loop
        T_solution = Function(V)
        C_solution = Function(V)

        for n in range(num_steps):
            # Solve heat equation
            solve(T_problem == T_rhs, T_solution, bc_T)
            Tn.assign(T_solution)

            # Update diffusion coefficient
            D_expr = project(D0 * exp(-Q / (R * (T_solution + 273.15))), V)

            # Define diffusion variational problem with updated D_expr
            F_C = (C_trial - Cn) / dt * w * dx + D_expr * dot(grad(C_trial), grad(w)) * dx
            C_problem, C_rhs = lhs(F_C), rhs(F_C)

            # Solve diffusion equation
            solve(C_problem == C_rhs, C_solution, bc_C)
            Cn.assign(C_solution)

            # Store results at selected intervals (adjusted for short test)
            if n % 15 == 0:
                times.append(n * dt)
                T_profiles.append(T_solution.vector().get_local())
                C_profiles.append(C_solution.vector().get_local())

        # Always store the final state
        times.append(T_end)
        T_profiles.append(T_solution.vector().get_local())
        C_profiles.append(C_solution.vector().get_local())
        
        # Calculate layer thickness based on concentration threshold
        coords = mesh.coordinates()
        sodium_concentration = C_solution.vector().get_local()
        layer_thickness = 0.0
        max_depth = 0.0

        for i in range(len(coords)):
            if sodium_concentration[i] >= concentration_threshold and coords[i][1] > max_depth:
                max_depth = coords[i][1]
        layer_thickness = max_depth

        # Save CSV results for sodium concentration
        csv_filename = f"CSV_Results/TEST_NaOH_{C_surface}M_Temp_{T_surface}C.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['X (m)', 'Y (m)', 'Sodium Concentration'])
            for i in range(len(coords)):
                writer.writerow([coords[i][0], coords[i][1], sodium_concentration[i]])
        print(f"Saved CSV result to {csv_filename}")

        # Save temperature profiles over time
        temp_csv_filename = f"Temperature_Profiles/TEST_TempProfile_NaOH_{C_surface}M_Temp_{T_surface}C.csv"
        with open(temp_csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time (s)'] + [f'Point {i}' for i in range(len(coords))])
            for idx, time in enumerate(times):
                row = [time] + list(T_profiles[idx])
                writer.writerow(row)
        print(f"Saved temperature profile to {temp_csv_filename}")

        # Append to summary report
        summary_data.append([T_surface, C_surface, layer_thickness * 1000])

        # Plot final sodium concentration
        import matplotlib.tri as tri
        triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], mesh.cells())
        plt.figure(figsize=(8, 6))
        plt.tricontourf(triangulation, C_solution.vector().get_local(), 50, cmap='viridis')
        plt.colorbar(label='Sodium Concentration')
        plt.title(f'TEST - NaOH: {C_surface}M, T_surface: {T_surface}°C\nLayer Thickness: {layer_thickness * 1000:.2f} mm')
        plt.xlabel('Length (m)')
        plt.ylabel('Thickness (m)')
        filename = f"Results/TEST_NaOH_{C_surface}M_Temp_{T_surface}C.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot to {filename}")

        # Create animation of concentration evolution
        fig, ax = plt.subplots(figsize=(8, 6))
        def update(frame):
            ax.clear()
            ax.tricontourf(triangulation, C_profiles[frame], 50, cmap='viridis', vmin=0, vmax=C_surface)
            ax.set_title(f'Time: {times[frame]:.1f} s')
            ax.set_xlabel('Length (m)')
            ax.set_ylabel('Thickness (m)')
            ax.set_xlim(0, length)
            ax.set_ylim(0, thickness)

        ani = animation.FuncAnimation(fig, update, frames=len(times), repeat=False)
        anim_filename = f"Animations/TEST_NaOH_{C_surface}M_Temp_{T_surface}C.gif"
        ani.save(anim_filename, writer='pillow', fps=5)
        plt.close()
        print(f"Saved animation to {anim_filename}")

# Save summary CSV
summary_filename = "Summary_Layer_Thickness_TEST.csv"
with open(summary_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Surface Temperature (°C)', 'NaOH Concentration (M)', 'Layer Thickness (mm)'])
    for row in summary_data:
        writer.writerow(row)

print(f"\nSummary report saved to {summary_filename}")
print("--- Short test completed successfully! ---")
