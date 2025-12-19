import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
import tempfile
from fsae_core import (
    SuspensionSolver, ForceSolver, LoadCaseGenerator, AnalysisTools,
    front_hp, rear_hp, VEHICLE_PARAMS,
    plot_corner, mirror_data, set_axes_proportional, plot_schematic_2d,
    dict_to_df, df_to_dict, calculate_damping_ratios
)

st.set_page_config(layout="wide", page_title="FSAE VD Suite")

#initialization
if 'front_df' not in st.session_state: st.session_state['front_df'] = dict_to_df(front_hp)
if 'rear_df' not in st.session_state: st.session_state['rear_df'] = dict_to_df(rear_hp)

st.title("FSAE Vehicle Dynamics Suite")

#SIDEBAR - vehicle setup
with st.sidebar:
    st.header("1. Vehicle Config")
    VEHICLE_PARAMS['mass_kg'] = st.number_input("Mass (kg)", 150.0, 400.0, 320.0, 5.0)
    VEHICLE_PARAMS['weight_dist'] = st.slider("Front Weight %", 0.40, 0.60, 0.50, 0.01)
    VEHICLE_PARAMS['cg_height'] = st.number_input("CG Height (mm)", 100.0, 500.0, 280.0, 5.0)
    st.subheader("Dimensions")
    VEHICLE_PARAMS['wheelbase'] = st.number_input("Wheelbase (mm)", 1300.0, 2000.0, 1650.0, 10.0)
    VEHICLE_PARAMS['track_width'] = st.number_input("Track Width (mm)", 1000.0, 1500.0, 1270.0, 10.0)
    VEHICLE_PARAMS['brake_bias'] = st.slider("Brake Bias (% Front)", 0.40, 0.80, 0.60, 0.01)
    VEHICLE_PARAMS['unsprung_mass_corner'] = st.number_input("Unsprung Mass/Corner (kg)", 5.0, 30.0, 15.0, 0.5)

    st.header("2. Suspension Targets")
    freq_f = st.number_input("Front Freq (Hz)", 1.0, 5.0, 3.0, 0.1)
    freq_r = st.number_input("Rear Freq (Hz)", 1.0, 5.0, 3.5, 0.1)
    VEHICLE_PARAMS['target_freq_front'] = freq_f
    VEHICLE_PARAMS['target_freq_rear'] = freq_r
    
    st.divider()
    bump_travel = st.slider("Bump (mm)", 10, 50, 25)
    droop_travel = st.slider("Droop (mm)", 10, 50, 25)
    
    if st.button("Download PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt=f"FSAE - VD Report", ln=1, align='C')
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 5, txt=f"Mass: {VEHICLE_PARAMS['mass_kg']}kg", ln=1)
        pdf.cell(0, 5, txt=f"CG: {VEHICLE_PARAMS['cg_height']}mm", ln=1)
        pdf.cell(0, 5, txt=f"Wheelbase: {VEHICLE_PARAMS['wheelbase']}mm", ln=1)
        pdf_out = pdf.output(dest='S').encode('latin-1')
        st.download_button(label="Save PDF", data=pdf_out, file_name="fsae_report.pdf", mime='application/pdf')

#solver exe
current_f_hp = df_to_dict(st.session_state['front_df'])
current_r_hp = df_to_dict(st.session_state['rear_df'])
f_solver = SuspensionSolver(current_f_hp)
r_solver = SuspensionSolver(current_r_hp)

viz_data = {'Front': f_solver.solve_heave(0), 'Rear': r_solver.solve_heave(0)}

#tabs
tab_geo, tab_kin, tab_steer, tab_anti, tab_damp, tab_viz, tab_loads = st.tabs([
    "Geometry Editor", "Kinematics", "Ackermann", "Anti-Dive/Squat", "Damping", "3D View", "FEA Loads"
])

# --- TAB 1: GEOMETRY ---
with tab_geo:
    c1, c2 = st.columns(2)
    with c1: 
        st.subheader("Front")
        st.session_state['front_df'] = st.data_editor(st.session_state['front_df'], height=300, key='ed_front')
    with c2: 
        st.subheader("Rear")
        st.session_state['rear_df'] = st.data_editor(st.session_state['rear_df'], height=300, key='ed_rear')
    st.divider()
    
    #2D schematics - front view
    st.subheader("Suspension Schematics (Front View)")
    fig_sch, (ax_sf, ax_sr) = plt.subplots(1, 2, figsize=(14, 6))
    if viz_data['Front']: plot_schematic_2d(ax_sf, current_f_hp, "Front Suspension", view='front')
    if viz_data['Rear']: plot_schematic_2d(ax_sr, current_r_hp, "Rear Suspension", view='front')
    st.pyplot(fig_sch)

# --- TAB 2: KINEMATICS ---
with tab_kin:
    heave = np.arange(-droop_travel, bump_travel + 1, 1)
    res_store = {'Front': [], 'Rear': []}
    
    for end, solver in [('Front', f_solver), ('Rear', r_solver)]:
        static = viz_data[end]
        if static:
            bc = solver.calculate_camber(static); bt = solver.calculate_toe(static)
            solver.init_guess = np.concatenate([static['upper_ball_joint'], static['lower_ball_joint'], static['tie_rod_upright']])
            for h in heave:
                r = solver.solve_heave(h)
                if r:
                    res_store[end].append([h, solver.calculate_camber(r) - bc, solver.calculate_toe(r) - bt, r['shock_len']])
                    solver.init_guess = np.concatenate([r['upper_ball_joint'], r['lower_ball_joint'], r['tie_rod_upright']])
                else: res_store[end].append([h, np.nan, np.nan, np.nan])
    
    f_data = np.array(res_store['Front']) if res_store['Front'] else np.zeros((len(heave), 4))
    r_data = np.array(res_store['Rear']) if res_store['Rear'] else np.zeros((len(heave), 4))
    
    k1, k2, k3 = st.columns(3)
    fig1, ax1 = plt.subplots()
    ax1.plot(f_data[:,0], f_data[:,1], 'b-', label='Front'); ax1.plot(r_data[:,0], r_data[:,1], 'r--', label='Rear')
    ax1.set_title("Camber Gain"); ax1.grid(True); ax1.legend(); k1.pyplot(fig1)
    
    fig2, ax2 = plt.subplots()
    ax2.plot(f_data[:,0], f_data[:,2], 'b-'); ax2.plot(r_data[:,0], r_data[:,2], 'r--')
    ax2.set_title("Bump Steer"); ax2.grid(True); k2.pyplot(fig2)
    
    mr_f_curve = np.abs(np.gradient(f_data[:,3], f_data[:,0]))
    mr_r_curve = np.abs(np.gradient(r_data[:,3], r_data[:,0]))
    
    fig3, ax3 = plt.subplots()
    ax3.plot(f_data[:,0], mr_f_curve, 'b-')
    ax3.plot(r_data[:,0], mr_r_curve, 'r--')
    ax3.set_title("Motion Ratio"); ax3.grid(True); k3.pyplot(fig3)
    
    st.divider()
    st.subheader("Spring Sizing")
    mr_f_avg = np.nanmean(mr_f_curve); mr_r_avg = np.nanmean(mr_r_curve)
    m_corner_f = ((VEHICLE_PARAMS['mass_kg']*VEHICLE_PARAMS['weight_dist'])/2) - VEHICLE_PARAMS['unsprung_mass_corner']
    m_corner_r = ((VEHICLE_PARAMS['mass_kg']*(1-VEHICLE_PARAMS['weight_dist']))/2) - VEHICLE_PARAMS['unsprung_mass_corner']
    k_f = (m_corner_f * (2*np.pi*freq_f)**2) / (mr_f_avg**2) / 1000
    k_r = (m_corner_r * (2*np.pi*freq_r)**2) / (mr_r_avg**2) / 1000
    sc1, sc2 = st.columns(2)
    sc1.metric("Front Spring Needed", f"{k_f:.1f} N/mm", f"MR: {mr_f_avg:.2f}")
    sc2.metric("Rear Spring Needed", f"{k_r:.1f} N/mm", f"MR: {mr_r_avg:.2f}")

# --- TAB 3: STEERING ---
with tab_steer:
    st.subheader("Ackermann Analysis")
    rack_travel = np.linspace(-15, 15, 20)
    steer_L = []; steer_R = []
    static_res = f_solver.solve_heave(0, steer_rack_y=0)
    static_toe = f_solver.calculate_toe(static_res) if static_res else 0.0
    if viz_data['Front']:
        f_solver.init_guess = np.concatenate([viz_data['Front']['upper_ball_joint'], viz_data['Front']['lower_ball_joint'], viz_data['Front']['tie_rod_upright']])
        for y in rack_travel:
            res = f_solver.solve_heave(0, steer_rack_y=y)
            if res:
                toe_raw = f_solver.calculate_toe(res)
                angle_L = toe_raw - static_toe
                res_mirror = f_solver.solve_heave(0, steer_rack_y=-y)
                angle_R = -(f_solver.calculate_toe(res_mirror) - static_toe) if res_mirror else -angle_L
                steer_L.append(angle_L); steer_R.append(angle_R)
    fig, ax = plt.subplots()
    if steer_L:
        inner_angs = []; outer_angs = []
        for l, r in zip(steer_L, steer_R):
            if l > 0: inner_angs.append(l); outer_angs.append(abs(r)) 
            elif r > 0: inner_angs.append(r); outer_angs.append(abs(l))
        if inner_angs:
            zipped = sorted(zip(inner_angs, outer_angs))
            i_plt, o_plt = zip(*zipped)
            ax.plot(i_plt, o_plt, 'b-', label='Actual Geometry')
        ax.plot([0, 20], [0, 20], 'k:', label="Parallel (100%)")
        ax.set_xlabel("Inner Wheel Angle (deg)"); ax.set_ylabel("Outer Wheel Angle (deg)"); ax.legend(); ax.grid(True)
        st.pyplot(fig)

# --- TAB 4: ANTI-DIVE ---
with tab_anti:
    st.subheader("Pitch Geometry (Side View)")
    c1, c2 = st.columns(2)
    # FRONT
    ic_f = AnalysisTools(current_f_hp).get_instant_center_side()
    anti_dive = AnalysisTools(current_f_hp).get_anti_percentage(ic_f, current_f_hp['wheel_center'][0], 'dive')
    c1.metric("Front Anti-Dive", f"{anti_dive:.1f} %")
    # REAR
    ic_r = AnalysisTools(current_r_hp).get_instant_center_side()
    anti_squat = AnalysisTools(current_r_hp).get_anti_percentage(ic_r, current_r_hp['wheel_center'][0], 'squat')
    c2.metric("Rear Anti-Squat", f"{anti_squat:.1f} %")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_schematic_2d(ax1, current_f_hp, "Front Side View", view='side')
    ax1.axhline(0, color='k', linewidth=2)
    if ic_f is not None:
        ax1.plot([ic_f[0], current_f_hp['wheel_center'][0]], [ic_f[1], 0], 'r--', label='Swing Arm')
        ax1.plot(ic_f[0], ic_f[1], 'ro')
        
    plot_schematic_2d(ax2, current_r_hp, "Rear Side View", view='side')
    ax2.axhline(0, color='k', linewidth=2)
    if ic_r is not None:
        ax2.plot([ic_r[0], current_r_hp['wheel_center'][0]], [ic_r[1], 0], 'r--', label='Swing Arm')
        ax2.plot(ic_r[0], ic_r[1], 'ro')
    st.pyplot(fig)

# --- TAB 5: DAMPING ---
with tab_damp:
    st.subheader("Damping Ratio Calculator")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info("Upload Shock Dyno CSV")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if not uploaded_file:
            st.warning("Using Generic Data")
            vels = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5])
            forces = np.array([0, 100, 180, 280, 350, 450])
            dyno_df = pd.DataFrame({'velocity_ms': vels, 'force_n': forces})
        else:
            dyno_df = pd.read_csv(uploaded_file)
        end_sel = st.radio("Select End", ["Front", "Rear"])
    with c2:
        if end_sel == "Front": mr, mass, spring = mr_f_avg, m_corner_f, k_f
        else: mr, mass, spring = mr_r_avg, m_corner_r, k_r
        res_df, c_crit = calculate_damping_ratios(dyno_df, mass, spring, mr)
        
        fig_d, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(12, 5))
        ax_d1.plot(res_df['velocity_ms']*1000, res_df['force_n'], 'b-o'); ax_d1.set_title("Force vs Vel")
        ax_d1.grid(True)
        ax_d2.plot(res_df['velocity_ms']*1000, res_df['Zeta'], 'g-o'); ax_d2.set_title("Damping Ratio")
        ax_d2.axhspan(0.65, 0.75, color='g', alpha=0.2)
        ax_d2.grid(True)
        st.pyplot(fig_d)

# --- TAB 6: 3D VIZ ---
with tab_viz:
    c1, c2 = st.columns([1, 4])
    with c1:
        elev = st.slider("Elevation", 0, 90, 30)
        azim = st.slider("Azimuth", -180, 180, 30)
    with c2:
        fig3d = plt.figure(figsize=(10, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')
        for end, data in viz_data.items():
            if data:
                c = 'b' if end == 'Front' else 'r'
                plot_corner(ax3d, data, c, VEHICLE_PARAMS)
                plot_corner(ax3d, mirror_data(data), c, VEHICLE_PARAMS)
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
        set_axes_proportional(ax3d); ax3d.view_init(elev=elev, azim=azim)
        st.pyplot(fig3d)

# --- TAB 7: LOADS ---
with tab_loads:
    st.header("FEA Loads")
    g_lat = st.slider("Lat G", 0.0, 2.5, 1.5); g_long = st.slider("Long G", -1.5, 2.0, 0.0)
    load_gen = LoadCaseGenerator(VEHICLE_PARAMS)
    loads = load_gen.get_loads(g_long, g_lat)
    if viz_data['Front']:
        st.write("Front Loads (N)")
        st.dataframe(pd.DataFrame([ForceSolver(viz_data['Front']).solve(loads['Front'])]).T)
    if viz_data['Rear']:
        st.write("Rear Loads (N)")
        st.dataframe(pd.DataFrame([ForceSolver(viz_data['Rear']).solve(loads['Rear'])]).T)