# File: simulation.py (Version 3.0 - Corrected and Robust Structure)

import fenics as fe
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def define_parameters():
    """Groups all simulation parameters into a single dictionary."""
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

def create_bcs(params, V):
    """Creates the Dirichlet boundary conditions."""
    class SurfaceBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[1], 0.0)

    surface = SurfaceBoundary()
    bc_T = fe.DirichletBC(V, fe.Constant(params["T_surface"]), surface)
    bc_C = fe.DirichletBC(V, fe.Constant(params["C_surface"]), surface)
    return bc_T, bc_C

def run_simulation(params, V, T_n, C_n, bc_T, bc_C):
    """
    Defines variational forms and runs the main time-stepping loop.
    """
    # 1. Define trial and test functions from the common FunctionSpace V
    T = fe.TrialFunction(V)
    C = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    w = fe.TestFunction(V)
    
    # Define the Function that will hold the changing diffusion coefficient
    D = fe.Function(V)

    # 2. Define the variational forms (equations)
    # Using T_n and C_n which are defined on the same FunctionSpace as T, v, etc.
    # This correctly avoids any mesh mismatches.
    dt = fe.Constant(params["dt"])
    F_T = (params["rho"] * params["Cp"] * T * v * fe.dx) + \
          (dt * params["k"] * fe.dot(fe.grad(T), fe.grad(v)) * fe.dx) - \
          (params["rho"] * params["Cp"] * T_n * v * fe.dx)
    
    a_T, L_T = fe.lhs(F_T), fe.rhs(F_T)

    # For diffusion, the LHS (a_C) changes every step because D changes.
    # We define the full form and will re-assemble 'a_C' inside the loop.
    F_C = (C * w * fe.dx) + (dt * D * fe.dot(fe.grad(C), fe.grad(w)) * fe.dx) - (C_n * w * fe.dx)
    a_C_form = fe.lhs(F_C)
    L_C_form = fe.rhs(F_C)

    # 3. Prepare for time-stepping
    T_solution = fe.Function(V)
    C_solution = fe.Function(V)
    num_steps = int(params["T_end"] / params["dt"])

    print("Starting simulation...")
    for n in range(num_steps):
        current_time = (n + 1) * params["dt"]

        # --- Solve Heat Equation ---
        fe.solve(a_T == L_T, T_solution, bc_T)
        
        # --- Update Diffusion Coefficient based on new temperature ---
        D_expr = fe.Expression('D0 * exp(-Q / (R * (T_k + 273.15)))',
                               degree=2, D0=params["D0"], Q=params["Q"], R=params["R"], T_k=T_solution)
        D.assign(fe.project(D_expr, V))

        # --- Solve Diffusion Equation ---
        # Re-assemble the LHS matrix because D has changed
        a_C = fe.assemble(a_C_form)
        L_C = fe.assemble(L_C_form)
        bc_C.apply(a_C, L_C)
        fe.solve(a_C, C_solution.vector(), L_C)

        # --- Update solutions for the next step ---
        T_n.assign(T_solution)
        C_n.assign(C_solution)

        if (n + 1) % 10 == 0:
            print(f"Solved up to t = {current_time:.0f} s")

    print("Simulation finished.")
    return C_solution

def plot_and_show(params, mesh, final_concentration):
    """Generates and displays the final plot."""
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
    params = define_parameters()

    # 2. Set up the core FEniCS objects in the main scope
    mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(params["length"], params["thickness"]), params["nx"], params["ny"])
    V = fe.FunctionSpace(mesh, 'P', 1)

    # 3. Create functions to hold the solutions from the previous time step (t_n)
    T_n = fe.interpolate(fe.Constant(params["T_initial"]), V)
    C_n = fe.interpolate(fe.Constant(params["C_initial"]), V)

    # 4. Create the boundary conditions
    bc_T, bc_C = create_bcs(params, V)

    # 5. Run the simulation by passing the necessary objects
    final_C = run_simulation(params, V, T_n, C_n, bc_T, bc_C)

    # 6. Generate the final plot
    plot_and_show(params, mesh, final_C)
