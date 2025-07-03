# File: simulation.py (Version 2.1 - Corrected and Refactored)

# Import necessary libraries
from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def define_parameters():
    """
    Groups all simulation parameters into a single dictionary.
    """
    params = {
        # --- Geometry and Mesh ---
        "length": 0.025, "thickness": 0.005,
        "nx": 50, "ny": 20,
        # --- Physics and Material Properties ---
        "rho": 4430.0, "Cp": 526.0, "k": 6.7,
        "D0": 1e-8, "Q": 100e3, "R": 8.314,
        # --- Time Stepping ---
        "dt": 60.0, "T_end": 3600.0,
        # --- Initial and Boundary Conditions ---
        "T_initial": 25.0, "C_initial": 0.0,
        "T_surface": 100.0, "C_surface": 5.0,
        # --- Plotting ---
        "plot_title": "Final Sodium Concentration Profile",
        "colorbar_label": "Sodium Concentration (Molar)"
    }
    return params

def setup_problem(params, placeholder_T_n, placeholder_C_n):
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

    # 3. Set up the variational problems using the placeholders
    T = TrialFunction(V)
    C = TrialFunction(V)
    v = TestFunction(V)
    w = TestFunction(V)
    D = Function(V) # Diffusion coefficient

    F_T = (params["rho"] * params["Cp"] * T * v * dx) + \
          (params["dt"] * params["k"] * dot(grad(T), grad(v)) * dx) - \
          (params["rho"] * params["Cp"] * placeholder_T_n * v * dx)
    
    F_C = (C * w * dx) + \
          (params["dt"] * D * dot(grad(C), grad(w)) * dx) - \
          (placeholder_C_n * w * dx)

    # We now need to assemble the LHS matrices here since they don't change
    a_T = assemble(lhs(F_T))
    a_C = assemble(lhs(F_C))
    
    # And we store the RHS forms which DO change
    L_T_form = rhs(F_T)
    L_C_form = rhs(F_C)

    problem_setup = {
        "mesh": mesh, "V": V, "D": D,
        "bc_T": bc_T, "bc_C": bc_C,
        "a_T": a_T, "L_T_form": L_T_form,
        "a_C": a_C, "L_C_form": L_C_form
    }
    return problem_setup

def run_simulation(params, problem_setup, placeholder_T_n, placeholder_C_n):
    """
    Runs the main time-stepping loop.
    """
    V = problem_setup["V"]
    D = problem_setup["D"]
    
    T_n = interpolate(Constant(params["T_initial"]), V)
    C_n = interpolate(Constant(params["C_initial"]), V)

    T_solution = Function(V)
    C_solution = Function(V)
    num_steps = int(params["T_end"] / params["dt"])

    print("Starting simulation...")
    for n in range(num_steps):
        current_time = (n + 1) * params["dt"]

        # Assemble the right-hand side vectors using the previous step's solutions
        L_T = assemble(problem_setup["L_T_form"].replace({placeholder_T_n: T_n}))
        L_C = assemble(problem_setup["L_C_form"].replace({placeholder_C_n: C_n}))
        
        # Apply boundary conditions to the vectors
        problem_setup["bc_T"].apply(L_T)
        problem_setup["bc_C"].apply(L_C)

        # Solve the linear systems
        solve(problem_setup["a_T"], T_solution.vector(), L_T)
        
        D_expr = Expression('D0 * exp(-Q / (R * (T_k + 273.15)))',
                            degree=2, D0=params["D0"], Q=params["Q"], R=params["R"], T_k=T_solution)
        D.assign(project(D_expr, V))

        # Re-assemble the diffusion LHS matrix because 'D' has changed
        a_C = assemble(problem_setup["a_C"].form)
        problem_setup["bc_C"].apply(a_C)

        solve(a_C, C_solution.vector(), L_C)

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
    # --- Main workflow ---
    # 1. Get all parameters
    simulation_params = define_parameters()

    # Create placeholder functions. These will be defined on the *correct* mesh later.
    # We initialize them as None for now.
    placeholder_T_n = None
    placeholder_C_n = None
    
    # A cleaner approach to defining forms and solving. We define UFL expression
    # for what will become our placeholders later. This avoids mesh-mismatch issues.
    V_dummy = FiniteElement("P", triangle, 1)
    placeholder_T_n = Coefficient(V_dummy)
    placeholder_C_n = Coefficient(V_dummy)

    # 2. Set up the finite element problem
    problem = setup_problem(simulation_params, placeholder_T_n, placeholder_C_n)

    # 3. Run the time-stepping simulation
    final_C = run_simulation(simulation_params, problem, placeholder_T_n, placeholder_C_n)

    # 4. Generate the final plot
    plot_and_show(simulation_params, problem["mesh"], final_C)
