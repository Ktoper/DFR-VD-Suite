import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fsae_core import (
    SuspensionSolver,
    ForceSolver,
    LoadCaseGenerator,
    dict_to_df,
    df_to_dict,
    calculate_damping_ratios,
    VEHICLE_PARAMS
)

st.set_page_config(layout="wide", page_title="FSAE VD Suite")

# ================= SESSION STATE =================
if 'front_df' not in st.session_state:
    st.session_state['front_df'] = dict_to_df({
        'upper_ball_joint': np.array([300, 600, 300]),
        'lower_ball_joint': np.array([300, 600, 150]),
        'tie_rod_upright': np.array([250, 580, 200]),
        'wheel_center': np.array([300, 630, 230])
    })

if 'rear_df' not in st.session_state:
    st.session_state['rear_df'] = dict_to_df({
        'upper_ball_joint': np.array([-1300, 600, 300]),
        'lower_ball_joint': np.array([-1300, 600, 150]),
        'tie_rod_upright': np.array([-1250, 580, 250]),
        'wheel_center': np.array([-1300, 630, 230])
    })

st.title("FSAE Vehicle Dynamics Suite")

# ================= SOLVERS =================
front_hp = df_to_dict(st.session_state['front_df'])
rear_hp = df_to_dict(st.session_state['rear_df'])

f_solver = SuspensionSolver(front_hp)
r_solver = SuspensionSolver(rear_hp)

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

    VEHICLE_PARAMS['mass_kg'] = st.number_input("Mass (kg)", 150.0, 400.0, 320.0)
    VEHICLE_PARAMS['weight_dist'] = st.slider("Front Weight %", 0.4, 0.6, 0.5)
    VEHICLE_PARAMS['cg_height'] = st.number_input("CG Height (mm)", 100.0, 500.0, 280.0)

    st.subheader("Dimensions")
    VEHICLE_PARAMS['wheelbase'] = st.number_input("Wheelbase (mm)", 1300.0, 2000.0, 1650.0)
    VEHICLE_PARAMS['track_width'] = st.number_input("Track Width (mm)", 1000.0, 1500.0, 1270.0)

# ====================================================
# TAB 2: GEOMETRY + 3D VIEW
# ====================================================
with tab_geo:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Front Geometry")
        st.session_state['front_df'] = st.data_editor(
            st.session_state['front_df'],
            height=300
        )

    with c2:
        st.subheader("Rear Geometry")
        st.session_state['rear_df'] = st.data_editor(
            st.session_state['rear_df'],
            height=300
        )

    st.divider()

    # -------- 3D VIEW --------
    st.subheader("3D Suspension View")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for end, data in viz_data.items():
        if data:
            color = 'b' if end == "Front" else 'r'
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    ax.scatter(v[0], v[1], v[2], color=color)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    st.pyplot(fig)

# ====================================================
# TAB 3: KINEMATICS
# ====================================================
with tab_kin:
    st.subheader("Camber & Toe vs Heave")

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
    st.subheader("Ackermann (Placeholder)")
    st.info("Detailed steering model can be expanded here.")

# ====================================================
# TAB 5: ANTI-DIVE / SQUAT
# ====================================================
with tab_anti:
    st.subheader("Anti Geometry (Coming Soon)")
    st.info("Add instant center / anti calculations here.")

# ====================================================
# TAB 6: DAMPING
# ====================================================
with tab_damp:
    st.subheader("Damping Ratio Tool")

    uploaded = st.file_uploader("Upload Shock CSV")

    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = pd.DataFrame({
            'velocity_ms': [0.05, 0.1, 0.2],
            'force_n': [100, 200, 350]
        })

    res, ccrit = calculate_damping_ratios(df, 300, 30, 1.0)

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

    st.subheader("Front")
    st.write(ForceSolver(viz_data['Front']).solve(loads['Front']))

    st.subheader("Rear")
    st.write(ForceSolver(viz_data['Rear']).solve(loads['Rear']))
