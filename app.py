import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fsae_core import (
    SuspensionSolver,
    ForceSolver,
    LoadCaseGenerator,
    AnalysisTools,
    plot_schematic_2d,
    dict_to_df,
    df_to_dict,
    calculate_damping_ratios,
    VEHICLE_PARAMS,
    front_hp,
    rear_hp,
    plot_corner,
    mirror_data,
    set_axes_proportional
)

st.set_page_config(layout="wide", page_title="FSAE VD Suite")

# ================= SESSION STATE =================
if 'front_df' not in st.session_state:
    st.session_state['front_df'] = dict_to_df(front_hp)

if 'rear_df' not in st.session_state:
    st.session_state['rear_df'] = dict_to_df(rear_hp)

st.title("FSAE Vehicle Dynamics Suite")

# ================= SOLVERS =================
current_f_hp = df_to_dict(st.session_state['front_df'])
current_r_hp = df_to_dict(st.session_state['rear_df'])

f_solver = SuspensionSolver(current_f_hp)
r_solver = SuspensionSolver(current_r_hp)

viz_data = {
    'Front': f_solver.solve_heave(0),
    'Rear': r_solver.solve_heave(0)
}

# ================= TABS =================
tab_config, tab_geo, tab_kin, tab_steer, tab_anti, tab_damp, tab_loads = st.tabs([
    "Vehicle Config",
    "Geometry Editor",
    "Kinematics",
    "Ackermann",
    "Anti-Dive/Squat",
    "Damping",
    "FEA Loads"
])

# ====================================================
# TAB 1: VEHICLE CONFIG
# ====================================================
with tab_config:
    st.header("Vehicle Setup")

    VEHICLE_PARAMS['mass_kg'] = st.number_input("Mass (kg)", 150.0, 400.0, VEHICLE_PARAMS['mass_kg'])
    VEHICLE_PARAMS['weight_dist'] = st.slider("Front Weight %", 0.4, 0.6, VEHICLE_PARAMS['weight_dist'])
    VEHICLE_PARAMS['cg_height'] = st.number_input("CG Height (mm)", 100.0, 500.0, VEHICLE_PARAMS['cg_height'])

    st.subheader("Dimensions")
    VEHICLE_PARAMS['wheelbase'] = st.number_input("Wheelbase (mm)", 1300.0, 2000.0, VEHICLE_PARAMS['wheelbase'])
    VEHICLE_PARAMS['track_width'] = st.number_input("Track Width (mm)", 1000.0, 1500.0, VEHICLE_PARAMS['track_width'])
    VEHICLE_PARAMS['brake_bias'] = st.slider("Brake Bias", 0.4, 0.8, VEHICLE_PARAMS['brake_bias'])

# ====================================================
# TAB 2: GEOMETRY + FULL 3D VIEW
# ====================================================
with tab_geo:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Front Geometry")
        st.session_state['front_df'] = st.data_editor(
            st.session_state['front_df'],
            height=400
        )

    with c2:
        st.subheader("Rear Geometry")
        st.session_state['rear_df'] = st.data_editor(
            st.session_state['rear_df'],
            height=400
        )

    st.divider()

    #----------------------------------------------
    st.subheader("3D Suspension View")

    c1, c2 = st.columns([1, 4])

    with c1:
        elev = st.slider("Elevation", 0, 90, 20, key="geo_elev")
        azim = st.slider("Azimuth", -180, 180, -60, key="geo_azim")

    with c2:
        fig3d = plt.figure(figsize=(10, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')

        # ✅ Ground plane
        xx, yy = np.meshgrid(
            np.linspace(-2000, 1000, 12),
            np.linspace(-1000, 1000, 12)
        )
        ax3d.plot_wireframe(xx, yy, np.zeros_like(xx), color='grey', alpha=0.1)

        # ✅ FULL vehicle rendering
        for end, data in viz_data.items():
            if data:
                color = 'b' if end == 'Front' else 'r'

                # Left side
                plot_corner(ax3d, data, color, VEHICLE_PARAMS)

                # Right side (mirrored)
                plot_corner(ax3d, mirror_data(data), color, VEHICLE_PARAMS)

        ax3d.set_title("FSAE Full Vehicle Model", fontsize=14)
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

        set_axes_proportional(ax3d)
        ax3d.view_init(elev=elev, azim=azim)

        st.pyplot(fig3d)

# ====================================================
# TAB 3: KINEMATICS
# ====================================================
with tab_kin:
    st.subheader("Camber Gain")

    heave = np.arange(-25, 26, 1)
    results = {'Front': [], 'Rear': []}

    for end, solver in [("Front", f_solver), ("Rear", r_solver)]:
        base = viz_data[end]
        bc = solver.calculate_camber(base)
        bt = solver.calculate_toe(base)

        for h in heave:
            r = solver.solve_heave(h)
            results[end].append([
                h,
                solver.calculate_camber(r) - bc,
                solver.calculate_toe(r) - bt,
                r['shock_len']
            ])

    f_data = np.array(results['Front'])
    r_data = np.array(results['Rear'])

    fig, ax = plt.subplots()
    ax.plot(f_data[:, 0], f_data[:, 1], label="Front")
    ax.plot(r_data[:, 0], r_data[:, 1], label="Rear")
    ax.set_title("Camber Gain")
    ax.grid()
    ax.legend()

    st.pyplot(fig)

# ====================================================
# TAB 4: ACKERMANN
# ====================================================
with tab_steer:
    st.subheader("Ackermann Analysis")

    rack_travel = np.linspace(-15, 15, 41)

    steer_L = []
    steer_R = []

    static_res = f_solver.solve_heave(0, steer_rack_y=0)

    if static_res:

        static_toe = f_solver.calculate_toe(static_res)

        # Improve convergence
        if viz_data['Front']:
            f_solver.init_guess = np.concatenate([
                viz_data['Front']['upper_ball_joint'],
                viz_data['Front']['lower_ball_joint'],
                viz_data['Front']['tie_rod_upright']
            ])

        for rack in rack_travel:

            # Left lock
            res_L = f_solver.solve_heave(
                0,
                steer_rack_y=rack
            )

            # Mirror approximation of opposite side
            res_R = f_solver.solve_heave(
                0,
                steer_rack_y=-rack
            )

            if res_L and res_R:

                left_angle = (
                    f_solver.calculate_toe(res_L)
                    - static_toe
                )

                right_angle = -(
                    f_solver.calculate_toe(res_R)
                    - static_toe
                )

                steer_L.append(left_angle)
                steer_R.append(right_angle)

    if steer_L:

        inner_angles = []
        outer_angles = []
        ackermann_pct = []

        wheelbase = VEHICLE_PARAMS['wheelbase']
        track = VEHICLE_PARAMS['track_width']

        for l, r in zip(steer_L, steer_R):

            # Ignore straight-ahead region
            if abs(l) < 0.25 and abs(r) < 0.25:
                continue

            inner = max(abs(l), abs(r))
            outer = min(abs(l), abs(r))

            inner_angles.append(inner)
            outer_angles.append(outer)

            try:

                ack = (
                    (
                        1 / np.tan(np.radians(outer))
                        -
                        1 / np.tan(np.radians(inner))
                    )
                    /
                    (track / wheelbase)
                ) * 100

                ackermann_pct.append(ack)

            except Exception:
                ackermann_pct.append(np.nan)

        # ------------------------------------
        # Plot 1: Outer vs Inner
        # ------------------------------------

        fig1, ax1 = plt.subplots(figsize=(6, 5))

        if inner_angles:

            pts = sorted(zip(inner_angles, outer_angles))
            inner_plot, outer_plot = zip(*pts)

            ax1.plot(
                inner_plot,
                outer_plot,
                linewidth=2,
                label="Actual Geometry"
            )

        # Parallel steering
        ax1.plot(
            [0, 25],
            [0, 25],
            'k:',
            label="Parallel Steering (0% Ackermann)"
        )

        ax1.set_xlabel("Inner Wheel Angle (deg)")
        ax1.set_ylabel("Outer Wheel Angle (deg)")
        ax1.set_title("Steering Geometry")
        ax1.grid(True)
        ax1.legend()

        st.pyplot(fig1)

        # ------------------------------------
        # Plot 2: Ackermann %
        # ------------------------------------

        fig2, ax2 = plt.subplots(figsize=(6, 5))

        ax2.plot(
            inner_angles,
            ackermann_pct,
            linewidth=2,
            color='tab:blue'
        )

        ax2.axhline(
            100,
            color='green',
            linestyle='--',
            label='Ideal Ackermann'
        )

        ax2.axhline(
            0,
            color='black',
            linestyle=':'
        )

        ax2.set_xlabel("Inner Wheel Angle (deg)")
        ax2.set_ylabel("Ackermann (%)")
        ax2.set_title("Ackermann Percentage")
        ax2.grid(True)
        ax2.legend()

        st.pyplot(fig2)

        # ------------------------------------
        # Summary metrics
        # ------------------------------------

        valid_ack = np.array(ackermann_pct)
        valid_ack = valid_ack[np.isfinite(valid_ack)]

        if len(valid_ack):

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Average Ackermann",
                f"{np.mean(valid_ack):.1f}%"
            )

            c2.metric(
                "Peak Ackermann",
                f"{np.max(valid_ack):.1f}%"
            )

            c3.metric(
                "Ackermann @ 10°",
                f"{np.interp(10, inner_angles, ackermann_pct):.1f}%"
            )

    else:
        st.warning("Unable to solve steering geometry.")
# ====================================================
# TAB 5: ANTI-DIVE / ANTI-SQUAT
# ====================================================
with tab_anti:

    st.subheader("Pitch Geometry Analysis")

    front_tools = AnalysisTools(current_f_hp)
    rear_tools = AnalysisTools(current_r_hp)

    ic_f = front_tools.get_instant_center_side()
    ic_r = rear_tools.get_instant_center_side()

    anti_dive = front_tools.get_anti_percentage(
        ic_f,
        current_f_hp['wheel_center'][0],
        'dive'
    )

    anti_squat = rear_tools.get_anti_percentage(
        ic_r,
        current_r_hp['wheel_center'][0],
        'squat'
    )

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Front Anti-Dive",
        f"{anti_dive:.1f}%"
    )

    c2.metric(
        "Rear Anti-Squat",
        f"{anti_squat:.1f}%"
    )

    if ic_f is not None:
        c3.metric(
            "Front IC Height",
            f"{ic_f[1]:.0f} mm"
        )

    if ic_r is not None:
        c4.metric(
            "Rear IC Height",
            f"{ic_r[1]:.0f} mm"
        )

    st.divider()

    # -----------------------------------------
    # Geometry Plots
    # -----------------------------------------

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # =================================================
    # FRONT
    # =================================================

    plot_schematic_2d(
        ax1,
        current_f_hp,
        "Front Side View",
        view='side'
    )

    ax1.axhline(
        0,
        color='black',
        linewidth=2
    )

    wc_x_f = current_f_hp['wheel_center'][0]

    if ic_f is not None:

        ax1.plot(
            ic_f[0],
            ic_f[1],
            'ro',
            markersize=8,
            label='Instant Center'
        )

        ax1.plot(
            [ic_f[0], wc_x_f],
            [ic_f[1], 0],
            'r--',
            linewidth=2,
            label='Swing Arm'
        )

        swing_angle = np.degrees(
            np.arctan2(
                -ic_f[1],
                wc_x_f - ic_f[0]
            )
        )

        ax1.text(
            0.05,
            0.95,
            f"IC = ({ic_f[0]:.0f}, {ic_f[1]:.0f}) mm\n"
            f"Swing Arm = {swing_angle:.1f}°\n"
            f"Anti-Dive = {anti_dive:.1f}%",
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )

    ax1.set_title("Front Suspension")
    ax1.grid(True)
    ax1.legend()

    # =================================================
    # REAR
    # =================================================

    plot_schematic_2d(
        ax2,
        current_r_hp,
        "Rear Side View",
        view='side'
    )

    ax2.axhline(
        0,
        color='black',
        linewidth=2
    )

    wc_x_r = current_r_hp['wheel_center'][0]

    if ic_r is not None:

        ax2.plot(
            ic_r[0],
            ic_r[1],
            'ro',
            markersize=8,
            label='Instant Center'
        )

        ax2.plot(
            [ic_r[0], wc_x_r],
            [ic_r[1], 0],
            'r--',
            linewidth=2,
            label='Swing Arm'
        )

        swing_angle = np.degrees(
            np.arctan2(
                -ic_r[1],
                wc_x_r - ic_r[0]
            )
        )

        ax2.text(
            0.05,
            0.95,
            f"IC = ({ic_r[0]:.0f}, {ic_r[1]:.0f}) mm\n"
            f"Swing Arm = {swing_angle:.1f}°\n"
            f"Anti-Squat = {anti_squat:.1f}%",
            transform=ax2.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )

    ax2.set_title("Rear Suspension")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    st.pyplot(fig)

    # -----------------------------------------
    # Classification
    # -----------------------------------------

    def classify(value):

        if value < 20:
            return "Low"

        elif value < 60:
            return "Moderate"

        elif value < 100:
            return "High"

        else:
            return "Pro"

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.info(
            f"Front Geometry Classification: "
            f"{classify(anti_dive)} Anti-Dive"
        )

    with c2:
        st.info(
            f"Rear Geometry Classification: "
            f"{classify(anti_squat)} Anti-Squat"
        )

# ====================================================
# TAB 6: DAMPING
# ====================================================
with tab_damp:
    st.subheader("Damping")

    uploaded = st.file_uploader("Upload Shock Dyno CSV")

    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = pd.DataFrame({
            'velocity_ms': [0.05, 0.1, 0.2],
            'force_n': [100, 200, 350]
        })

    res, _ = calculate_damping_ratios(df, 300, 30, 1.0)

    fig, ax = plt.subplots()
    ax.plot(res['velocity_ms'], res['Zeta'])
    ax.set_title("Damping Ratio")
    ax.grid()

    st.pyplot(fig)

# ====================================================
# TAB 7: LOADS
# ====================================================
with tab_loads:
    st.header("FEA Loads")

    g_lat = st.slider("Lat G", 0.0, 2.5, 1.5)
    g_long = st.slider("Long G", -1.5, 2.0, 0.0)

    load_gen = LoadCaseGenerator(VEHICLE_PARAMS)
    loads = load_gen.get_loads(g_long, g_lat)

    st.write("Front Loads")
    st.write(ForceSolver(viz_data['Front']).solve(loads['Front']))

    st.write("Rear Loads")
    st.write(ForceSolver(viz_data['Rear']).solve(loads['Rear']))
