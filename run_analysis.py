import matplotlib.pyplot as plt
import numpy as np
from fsae_core import (
    SuspensionSolver, ForceSolver, LoadCaseGenerator,
    front_hp, rear_hp, VEHICLE_PARAMS,
    plot_corner, mirror_data, set_axes_proportional, export_fea_loads
)

def run():
    print(f"{'='*60}")
    print("FSAE VEHICLE DYNAMICS SUITE")
    print(f"{'='*60}")

    # 1. SETUP
    f_solver = SuspensionSolver(front_hp)
    r_solver = SuspensionSolver(rear_hp)
    load_gen = LoadCaseGenerator(VEHICLE_PARAMS)

    # 2. FEA LOADS (Static)
    print("\n[1/3] Calculating FEA Loads (1.5G Braking)...")
    loads = load_gen.get_loads(1.5, 0.0)
    f_res = ForceSolver(f_solver.solve_heave(0)).solve(loads['Front'])
    r_res = ForceSolver(r_solver.solve_heave(0)).solve(loads['Rear'])
    export_fea_loads(f_res, r_res)

    # 3. KINEMATIC SWEEP (init. "warm start")
    print("\n[2/3] Running Suspension Sweep...")
    heave_up = np.arange(0, 26, 1); heave_down = np.arange(-1, -26, -1)
    res_store = {'Front': {'heave':[], 'camber':[], 'toe':[], 'shock':[]}, 'Rear': {'heave':[], 'camber':[], 'toe':[], 'shock':[]}}
    viz_data = {'Front': None, 'Rear': None}

    for end, solver in [('Front', f_solver), ('Rear', r_solver)]:
        static = solver.solve_heave(0)
        viz_data[end] = static
        bc = solver.calculate_camber(static); bt = solver.calculate_toe(static)
        
        def run_sweep(vals):
            solver.init_guess = np.concatenate([solver.hp['ubj'], solver.hp['lbj'], solver.hp['tie_rod_outer']])
            for h in vals:
                r = solver.solve_heave(h)
                if r:
                    res_store[end]['heave'].append(h)
                    res_store[end]['camber'].append(solver.calculate_camber(r)-bc)
                    res_store[end]['toe'].append(solver.calculate_toe(r)-bt)
                    res_store[end]['shock'].append(r['shock_len'])
                    solver.init_guess = np.concatenate([r['ubj'], r['lbj'], r['tro']])
                else:
                    res_store[end]['heave'].append(h); res_store[end]['camber'].append(np.nan)
        run_sweep(heave_up); run_sweep(heave_down)
        
        if res_store[end]['heave']:
            zipped = sorted(zip(res_store[end]['heave'], res_store[end]['camber'], res_store[end]['toe'], res_store[end]['shock']))
            h, c, t, s = zip(*zipped)
            res_store[end]['heave'] = list(h); res_store[end]['camber'] = list(c); res_store[end]['toe'] = list(t); res_store[end]['shock'] = list(s)

    # 4. PLOTTING
    print("\n[3/3] Generating Visualizations...")
    fig = plt.figure(figsize=(18, 12))
    
    # graphs
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(res_store['Front']['heave'], res_store['Front']['camber'], 'b-', label='Front')
    ax1.plot(res_store['Rear']['heave'], res_store['Rear']['camber'], 'r--', label='Rear')
    ax1.set_title('Camber Gain (deg)'); ax1.grid(True); ax1.legend()
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(res_store['Front']['heave'], res_store['Front']['toe'], 'b-')
    ax2.plot(res_store['Rear']['heave'], res_store['Rear']['toe'], 'r--')
    ax2.set_title('Bump Steer (deg)'); ax2.grid(True)
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(res_store['Front']['heave'], np.abs(np.gradient(res_store['Front']['shock'], res_store['Front']['heave'])), 'b-')
    ax3.plot(res_store['Rear']['heave'], np.abs(np.gradient(res_store['Rear']['shock'], res_store['Rear']['heave'])), 'r--')
    ax3.set_title('Motion Ratio'); ax3.grid(True)

    # 3D
    ax3d = fig.add_subplot(2, 1, 2, projection='3d')
    xx, yy = np.meshgrid(np.linspace(-2000, 1000, 12), np.linspace(-1000, 1000, 12))
    ax3d.plot_wireframe(xx, yy, np.zeros_like(xx), color='grey', alpha=0.1)
    for end, data in viz_data.items():
        c = 'b' if end == 'Front' else 'r'
        plot_corner(ax3d, data, c, VEHICLE_PARAMS)
        plot_corner(ax3d, mirror_data(data), c, VEHICLE_PARAMS)
    
    ax3d.set_title("FSAE Full Vehicle Model (Orthographic)", fontsize=16)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    set_axes_proportional(ax3d); ax3d.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()