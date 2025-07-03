# File: simulation.py (Version 2.0 - Refactored for Quality)

# Import necessary libraries
from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def define_parameters():
    """
    Groups all simulation parameters into a single dictionary.
    This is the only place you need to edit to change simulation settings.
    """
    params = {
        # --- Geometry and Mesh ---
        "length": 0.025,      # m
        "thickness": 0.005,   # m
        "nx": 50,             # Number of elements in x-direction
        "ny": 20,             # Number of elements in y-direction

        # --- Physics and Material Properties ---
        "rho": 4430.0,      # Density (kg/m^3)
        "Cp": 526.0,        # Specific Heat (J/kg.K)
        "k": 6.7,           # Thermal conductivity (W/m.K)
        "D0": 1e-8,         # Pre-exponential factor for diffusion (m^2/s)
        "Q": 100e3,         # Activation energy for diffusion (J/mol)
        "R": 8.314,         # Ideal gas constant (J/mol.K)

        # --- Time Stepping ---
        "dt": 60.0,         # Time step (s)
        "T_end": 3600.0,    # Total simulation time (s)

        # --- Initial and Boundary Conditions ---
        "T_initial": 25.0,    # Initial temperature in Celsius
        "C_initial": 0.0,     # Initial concentration in Molar
        "T_surface": 100.0,   # Surface temperature in Celsius
        "C_surface": 5.0,     # Surface concentration in Molar
        
        # --- Plotting ---
        "plot_title": "Final Sodium Concentration Profile",
        "colorbar_label": "Sodium Concentration (Molar)"
    }
    return params

def setup_problem(params):
    """
    Sets up the mesh, function spaces, boundary conditions, and variational forms.
    """
    # 1. Create mesh and define function space
    mesh = RectangleMesh(Point(0, 0), Point(params["length"], params["thickness"]), params["nx"], params["ny"])
    V = FunctionSpace(mesh, 'P', 1)

    # 2. Define boundary conditions
    class SurfaceBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0)

    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    surface = SurfaceBoundary()
    surface.mark(boundary_markers, 1)

    bc_T = DirichletBC(V, Constant(params["T_surface"]), boundary_markers, 1)
    bc_C = DirichletBC(V, Constant(params["C_surface"]), boundary_markers, 1)

    # 3. Set up the variational problems
    T = TrialFunction(V)
    C = TrialFunction(V)
    v = TestFunction(V)
    w = TestFunction(V)
    D = Function(V) # Diffusion coefficient as a Function

    F_T = (params["rho"] * params["Cp"] * T * v * dx) + \
          (params["dt"] * params["k"] * dot(grad(T), grad(v)) * dx) - \
          (params["rho"] * params["Cp"] * placeholder_T_n * v * dx)
    
    F_C = (C * w * dx) + \
          (params["dt"] * D * dot(grad(C), grad(w)) * dx) - \
          (placeholder_C_n * w * dx)

    a_T, L_T_form = lhs(F_T), rhs(F_T)
    a_C, L_C_form = lhs(F_C), rhs(F_C)

    # Return a dictionary containing all the necessary setup objects
    problem_setup = {
        "mesh": mesh, "V": V, "D": D,
        "bc_T": bc_T, "bc_C": bc_C,
        "a_T": a_T, "L_T_form": L_T_form,
        "a_C": a_C, "L_C_form": L_C_form
    }
    return problem_setup

def run_simulation(params, problem_setup):
    """
    Runs the main time-stepping loop.
    """
    # Unpack the problem setup dictionary
    V = problem_setup["V"]
    D = problem_setup["D"]
    
    # 1. Define initial conditions
    T_n = interpolate(Constant(params["T_initial"]), V)
    C_n = interpolate(Constant(params["C_initial"]), V)

    # 2. Prepare for time-stepping
    T_solution = Function(V)
    C_solution = Function(V)
    num_steps = int(params["T_end"] / params["dt"])

    print("Starting simulation...")
    for n in range(num_steps):
        current_time = (n + 1) * params["dt"]

        # Assemble the right-hand side vectors using the previous step's solutions
        L_T = assemble(problem_setup["L_T_form"].replace({placeholder_T_n: T_n}))
        L_C = assemble(problem_setup["L_C_form"].replace({placeholder_C_n: C_n}))
        
        # Solve Heat Equation
        solve(problem_setup["a_T"], T_solution.vector(), L_T, problem_setup["bc_T"])
        
        # Update Diffusion Coefficient
        D_expr = Expression('D0 * exp(-Q / (R * (T_k + 273.15)))',
                            degree=2, D0=params["D0"], Q=params["Q"], R=params["R"], T_k=T_solution)
        D.assign(project(D_expr, V))

        # Solve Diffusion Equation
        solve(problem_setup["a_C"], C_solution.vector(), L_C, problem_setup["bc_C"])

        # Update previous solutions for the next loop
        T_n.assign(T_solution)
        C_n.assign(C_solution)

        if (n + 1) % 10 == 0:
            print(f"Solved up to t = {current_time:.0f} s")

    print("Simulation finished.")
    return C_solution

def plot_and_show(params, mesh, final_concentration):
    """
    Generates and displays the final plot.
    """
    print("Plotting final concentration.")
    coords = mesh.coordinates()
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], mesh.cells())
    c_values = final_concentration.vector().get_local()

    plt.figure(figsize=(10, 5))
    contour = plt.tricontourf(triangulation, c_values, levels=50, cmap='viridis')
    plt.colorbar(contour, label=params["colorbar_label"])
    plt.title(f'{params["plot_title"]} at t = {params["T_end"]} s')
    plt.xlabel('Length (m)')
    plt.ylabel('Thickness (m)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    # Define placeholder Functions for the variational forms
    # This is a robust way to handle forms that depend on previous solutions
    V_placeholder = FunctionSpace(RectangleMesh(Point(0,0), Point(1,1), 1, 1), 'P', 1)
    placeholder_T_n = Function(V_placeholder)
    placeholder_C_n = Function(V_placeholder)

    # --- Main workflow ---
    # 1. Get all parameters
    simulation_params = define_parameters()

    # 2. Set up the finite element problem
    problem = setup_problem(simulation_params)

    # 3. Run the time-stepping simulation
    final_C = run_simulation(simulation_params, problem)

    # 4. Generate the final plot
    plot_and_show(simulation_params, problem["mesh"], final_C)
