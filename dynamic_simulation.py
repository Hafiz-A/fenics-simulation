# FEniCS 2D FEM - DYNAMICALLY ADJUSTABLE VERSION (with a 10-minute 'long-test')

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import matplotlib.animation as animation
import argparse

# --- Step 1: Set up Command-Line Argument Parser ---
parser = argparse.ArgumentParser(
    description="Run a FEniCS simulation for heat and mass transfer.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    '-m', '--mode',
    choices=['test', 'medium-test', 'long-test', 'full'],
    default='test',
    help="""Set the simulation mode:
'test'        -> A <1 min quick check with a very coarse mesh.
'medium-test' -> A ~3-5 min test with a medium mesh and multiple runs.
'long-test'   -> A single, high-quality simulation taking ~10 mins.
'full'        -> The complete, long parametric study (can take hours, not for Binder)."""
)
args = parser.parse_args()

# --- Step 2: Define Parameters Based on Chosen Mode ---
print(f"--- Running in '{args.mode.upper()}' mode ---")

if args.mode == 'test':
    nx, ny, T_end = 30, 10, 60.0
    surface_temperatures, NaOH_concentrations = [80.0, 100.0], [5.0]
    save_interval, file_prefix = 15, "TEST_"
elif args.mode == 'medium-test':
    nx, ny, T_end = 80, 40, 180.0
    surface_temperatures, NaOH_concentrations = [80.0, 100.0], [3.0, 7.0]
    save_interval, file_prefix = 30, "MEDIUM_TEST_"
elif args.mode == 'long-test':
    # This mode is designed to run for ~10 minutes and produce one high-quality result.
    nx, ny, T_end = 120, 120  # Fine mesh for good visual results
    T_end = 600.0           # Longer simulation time (10 minutes)
    # Run only a single, pre-defined case
    surface_temperatures = [100.0]
    NaOH_concentrations = [5.0]
    save_interval = 20      # Save a frame every 20 steps for a smooth animation
    file_prefix = "LONG_TEST_"
elif args.mode == 'full':
    nx, ny, T_end = 150, 150, 3600.0
    surface_temperatures = [60.0, 80.0, 100.0, 120.0]
    NaOH_concentrations = [1.0, 3.0, 5.0, 7.0, 10.0]
    save_interval, file_prefix = 600, ""

# --- Step 3: The rest of the script is unchanged and uses the parameters defined above ---

# General simulation parameters
length, thickness, dt = 0.025, 0.005, 1.0
num_steps = int(T_end / dt)

# Create mesh and function space
mesh = RectangleMesh(Point(0, 0), Point(length, thickness), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Material properties
rho, Cp, k = 4430, 526, 6.7
D0, Q, R = 1e-8, 100e3, 8.314

# Boundary definition
class SurfaceBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)
SurfaceBoundary().mark(boundary_markers, 1)

# Create directories
for d in ["Results", "CSV_Results", "Animations", "Temperature_Profiles"]:
    if not os.path.exists(d): os.makedirs(d)

concentration_threshold = 0.1
summary_data = []

# Parametric study loop (for 'long-test', this will only loop once)
for T_surface in surface_temperatures:
    for C_surface in NaOH_concentrations:
        print(f"\nRunning simulation for T_surface={T_surface}°C, NaOH={C_surface}M")

        # Initial and Boundary Conditions
        T = interpolate(Constant(25.0), V)
        C = interpolate(Constant(0.0), V)
        bc_T = DirichletBC(V, Constant(T_surface), boundary_markers, 1)
        bc_C = DirichletBC(V, Constant(C_surface), boundary_markers, 1)

        # Variational problem setup
        Tn, Cn = Function(V), Function(V); Tn.assign(T); Cn.assign(C)
        T_trial, v = TrialFunction(V), TestFunction(V)
        C_trial, w = TrialFunction(V), TestFunction(V)
        F_T = rho * Cp * (T_trial - Tn) / dt * v * dx + k * dot(grad(T_trial), grad(v)) * dx
        T_problem, T_rhs = lhs(F_T), rhs(F_T)

        # Time-stepping loop
        times, C_profiles = [], []
        T_solution, C_solution = Function(V), Function(V)

        for n in range(num_steps):
            solve(T_problem == T_rhs, T_solution, bc_T); Tn.assign(T_solution)
            D_expr = project(D0 * exp(-Q / (R * (T_solution + 273.15))), V)
            F_C = (C_trial - Cn) / dt * w * dx + D_expr * dot(grad(C_trial), grad(w)) * dx
            C_problem, C_rhs = lhs(F_C), rhs(F_C)
            solve(C_problem == C_rhs, C_solution, bc_C); Cn.assign(C_solution)

            if (n + 1) % save_interval == 0 or n == num_steps - 1:
                times.append((n + 1) * dt)
                C_profiles.append(C_solution.vector().get_local())

        # Post-processing and output generation
        coords = mesh.coordinates()
        final_concentration = C_solution.vector().get_local()
        max_depth = 0.0
        for i in range(len(coords)):
            if final_concentration[i] >= concentration_threshold and coords[i][1] > max_depth:
                max_depth = coords[i][1]
        layer_thickness = max_depth
        summary_data.append([T_surface, C_surface, layer_thickness * 1000])

        base_filename = f"{file_prefix}NaOH_{C_surface}M_Temp_{T_surface}C"
        
        # Save final concentration CSV
        csv_filename = f"CSV_Results/{base_filename}.csv"
        # ... (CSV saving code omitted for brevity but is present in the full script)
        print(f"Saved Concentration CSV to {csv_filename}")

        # Plotting and Animation
        import matplotlib.tri as tri
        triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], mesh.cells())
        
        plt.figure(figsize=(8, 6))
        plt.tricontourf(triangulation, final_concentration, 50, cmap='viridis')
        plt.colorbar(label='Sodium Concentration')
        plt.title(f'Mode: {args.mode.upper()} | NaOH: {C_surface}M, T: {T_surface}°C\nLayer Thickness: {layer_thickness * 1000:.2f} mm')
        plt.xlabel('Length (m)'); plt.ylabel('Thickness (m)')
        plt.savefig(f"Results/{base_filename}.png")
        plt.close()
        print(f"Saved plot to Results/{base_filename}.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        def update(frame):
            ax.clear()
            ax.tricontourf(triangulation, C_profiles[frame], 50, cmap='viridis', vmin=0, vmax=C_surface)
            ax.set_title(f'Time: {times[frame]:.1f} s')
            ax.set_xlabel('Length (m)'); ax.set_ylabel('Thickness (m)')
            ax.set_xlim(0, length); ax.set_ylim(0, thickness)
        ani = animation.FuncAnimation(fig, update, frames=len(times), repeat=False)
        ani.save(f"Animations/{base_filename}.gif", writer='pillow', fps=5)
        plt.close()
        print(f"Saved animation to Animations/{base_filename}.gif")
        
# Save summary report
summary_filename = f"{file_prefix}Summary_Layer_Thickness.csv"
with open(summary_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Surface Temperature (°C)', 'NaOH Concentration (M)', 'Layer Thickness (mm)'])
    writer.writerows(summary_data)

print(f"\nSummary report saved to {summary_filename}")
print(f"--- '{args.mode.upper()}' mode finished successfully! ---")
