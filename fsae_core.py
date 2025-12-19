import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# ============ VEHICLE DATA INPUT ============

VEHICLE_PARAMS = {
    'mass_kg': 320.0, 'cg_height': 280.0, 'wheelbase': 1650.0, 
    'track_width': 1270.0, 'weight_dist': 0.50,
    'tire_radius': 228.6, 'tire_width': 205.0,
    'brake_bias': 0.60,
    'unsprung_mass_corner': 15.0,
    'target_freq_front': 3.0, 'target_freq_rear': 3.5
}

front_hp = {
    'upper_wishbone_front': np.array([384.746, 265.190, 221.546]),
    'upper_wishbone_rear':  np.array([202.091, 275.030, 222.985]),
    'lower_wishbone_front': np.array([386.760, 193.690, 93.533]),
    'lower_wishbone_rear':  np.array([203.162, 218.530, 87.556]),
    'tie_rod_chassis':      np.array([241.296, 217.871, 121.199]),
    'upper_ball_joint':     np.array([296.013, 594.999, 297.614]),
    'lower_ball_joint':     np.array([306.743, 600.332, 144.281]),
    'tie_rod_upright':      np.array([223.932, 584.216, 180.706]),
    'wheel_center':         np.array([300.983, 627.022, 228.461]),
    'pushrod_upright_mount':  np.array([302.292, 562.587, 164.418]), 
    'rocker_pivot_point':     np.array([310.914, 245.000, 580.859]), 
    'rocker_axis_definition': np.array([410.892, 245.000, 578.789]), 
    'pushrod_rocker_mount':   np.array([311.505, 291.997, 609.424]),    
    'shock_rocker_mount':     np.array([311.999, 228.425, 633.291]), 
    'shock_chassis_mount':    np.array([310.914, 62.500, 625.000]), 
}

rear_hp = {
    'upper_wishbone_front': np.array([-1216.930, 300.000, 280.470]),
    'upper_wishbone_rear':  np.array([-1466.820, 300.000, 287.670]),
    'lower_wishbone_front': np.array([-1224.490, 280.000, 121.770]),
    'lower_wishbone_rear':  np.array([-1474.390, 280.000, 128.970]),
    'tie_rod_chassis':      np.array([-1220.710, 295.430, 238.850]),
    'upper_ball_joint':     np.array([-1352.771, 598.432, 346.443]),
    'lower_ball_joint':     np.array([-1351.393, 602.249, 171.490]),
    'tie_rod_upright':      np.array([-1263.639, 599.509, 297.755]),
    'wheel_center':         np.array([-1350.000, 630.013, 228.546]),
    'pushrod_upright_mount':  np.array([-1348.963, 552.751, 168.683]), 
    'rocker_pivot_point':     np.array([-1426.258, 260.411, 336.729]), 
    'rocker_axis_definition': np.array([-1522.266, 270.886, 310.792]), 
    'pushrod_rocker_mount':   np.array([-1428.888, 288.292, 357.722]),    
    'shock_rocker_mount':     np.array([-1445.120, 218.736, 389.714]), 
    'shock_chassis_mount':    np.array([-1451.099, 50.000, 343.698]), 
}

# ============ HELPER FUNCS ============

def dict_to_df(hp_dict):
    data = []
    for key, val in hp_dict.items():
        data.append({'Point': key, 'X': val[0], 'Y': val[1], 'Z': val[2]})
    return pd.DataFrame(data).set_index('Point')

def df_to_dict(df):
    hp_new = {}
    for index, row in df.iterrows():
        hp_new[index] = np.array([row['X'], row['Y'], row['Z']])
    return hp_new

def calculate_damping_ratios(dyno_df, corner_mass, spring_rate, motion_ratio):
    k_wheel_npmm = spring_rate * (motion_ratio**2)
    k_wheel_npm = k_wheel_npmm * 1000
    c_crit = 2 * np.sqrt(k_wheel_npm * corner_mass)
    results = dyno_df.copy()
    results['Cs'] = results['force_n'] / results['velocity_ms'].replace(0, 1e-9)
    results['Cw'] = results['Cs'] * (motion_ratio**2)
    results['Zeta'] = results['Cw'] / c_crit
    return results, c_crit

# ============ SOLVER CLASSES ============

class RigidBodyHelper:
    def __init__(self, ubj, lbj, tro, point_to_track):
        self.local_pos = self.global_to_local(ubj, lbj, tro, point_to_track)
    def get_basis(self, ubj, lbj, tro):
        origin = lbj
        vz = ubj - lbj; vz /= np.linalg.norm(vz)
        v_tro = tro - lbj; vy = np.cross(vz, v_tro); vy /= np.linalg.norm(vy)
        vx = np.cross(vy, vz); vx /= np.linalg.norm(vx)
        return origin, np.vstack([vx, vy, vz]).T
    def global_to_local(self, ubj, lbj, tro, pt):
        origin, R = self.get_basis(ubj, lbj, tro)
        return np.linalg.inv(R) @ (pt - origin)
    def get_new_position(self, new_ubj, new_lbj, new_tro):
        origin, R = self.get_basis(new_ubj, new_lbj, new_tro)
        return origin + (R @ self.local_pos)

class ActuationSolver:
    def __init__(self, hp):
        self.hp = hp
        self.rod_len = np.linalg.norm(hp['pushrod_upright_mount'] - hp['pushrod_rocker_mount'])
        self.pivot = hp['rocker_pivot_point']
        axis_vec = hp['rocker_axis_definition'] - hp['rocker_pivot_point']
        self.axis = axis_vec / np.linalg.norm(axis_vec)
        self.v_piv_to_pr = hp['pushrod_rocker_mount'] - self.pivot
        self.v_piv_to_shock = hp['shock_rocker_mount'] - self.pivot
    def rotate_vec(self, v, axis, theta):
        return (v * np.cos(theta) + np.cross(axis, v) * np.sin(theta) + axis * np.dot(axis, v) * (1 - np.cos(theta)))
    def solve(self, current_pr_upright_pos):
        def residuals(x):
            theta = x[0]
            curr_pr_rocker = self.pivot + self.rotate_vec(self.v_piv_to_pr, self.axis, theta)
            return np.linalg.norm(curr_pr_rocker - current_pr_upright_pos) - self.rod_len
        sol = root(residuals, [0.0], method='lm')
        if not sol.success: return None, None
        theta = sol.x[0]
        curr_pr_rocker = self.pivot + self.rotate_vec(self.v_piv_to_pr, self.axis, theta)
        curr_shock_rocker = self.pivot + self.rotate_vec(self.v_piv_to_shock, self.axis, theta)
        shock_len = np.linalg.norm(curr_shock_rocker - self.hp['shock_chassis_mount'])
        return shock_len, {'pushrod_rocker_mount': curr_pr_rocker, 'shock_rocker_mount': curr_shock_rocker}

class SuspensionSolver:
    def __init__(self, hardpoints):
        self.hp = hardpoints
        self.actuator = ActuationSolver(hardpoints)
        p = self.hp
        self.uca_rad = self.dist_pt_line(p['upper_ball_joint'], p['upper_wishbone_front'], p['upper_wishbone_rear'])
        self.lca_rad = self.dist_pt_line(p['lower_ball_joint'], p['lower_wishbone_front'], p['lower_wishbone_rear'])
        self.tr_len = np.linalg.norm(p['tie_rod_upright'] - p['tie_rod_chassis'])
        self.d_ul = np.linalg.norm(p['upper_ball_joint'] - p['lower_ball_joint'])
        self.d_ut = np.linalg.norm(p['upper_ball_joint'] - p['tie_rod_upright'])
        self.d_lt = np.linalg.norm(p['lower_ball_joint'] - p['tie_rod_upright'])
        
        self.spindle_tracker = RigidBodyHelper(p['upper_ball_joint'], p['lower_ball_joint'], p['tie_rod_upright'], p['wheel_center'])
        spindle_end_point = p['wheel_center'] + np.array([0, 500, 0])
        self.axis_tracker = RigidBodyHelper(p['upper_ball_joint'], p['lower_ball_joint'], p['tie_rod_upright'], spindle_end_point)
        
        self.pr_tracker = RigidBodyHelper(p['upper_ball_joint'], p['lower_ball_joint'], p['tie_rod_upright'], p['pushrod_upright_mount'])
        self.init_guess = np.concatenate([p['upper_ball_joint'], p['lower_ball_joint'], p['tie_rod_upright']])
    
    def dist_pt_line(self, pt, a, b):
        return np.linalg.norm(np.cross(pt - a, b - a)) / np.linalg.norm(b - a)
    
    def solve_heave(self, heave_mm, steer_rack_y=0):
        p = self.hp
        target_z = p['lower_ball_joint'][2] + heave_mm
        tri_current = p['tie_rod_chassis'] + np.array([0, steer_rack_y, 0])
        def residuals(vars):
            u, l, t = vars[0:3], vars[3:6], vars[6:9]
            res = []
            res.append(self.dist_pt_line(u, p['upper_wishbone_front'], p['upper_wishbone_rear']) - self.uca_rad)
            res.append(self.dist_pt_line(l, p['lower_wishbone_front'], p['lower_wishbone_rear']) - self.lca_rad)
            u_ax = (p['upper_wishbone_rear'] - p['upper_wishbone_front']) / np.linalg.norm(p['upper_wishbone_rear'] - p['upper_wishbone_front'])
            l_ax = (p['lower_wishbone_rear'] - p['lower_wishbone_front']) / np.linalg.norm(p['lower_wishbone_rear'] - p['lower_wishbone_front'])
            res.append(np.dot(u - p['upper_wishbone_front'], u_ax) - np.dot(p['upper_ball_joint'] - p['upper_wishbone_front'], u_ax))
            res.append(np.dot(l - p['lower_wishbone_front'], l_ax) - np.dot(p['lower_ball_joint'] - p['lower_wishbone_front'], l_ax))
            res.append(np.linalg.norm(t - tri_current) - self.tr_len)
            res.append(np.linalg.norm(u - l) - self.d_ul)
            res.append(np.linalg.norm(u - t) - self.d_ut)
            res.append(np.linalg.norm(l - t) - self.d_lt)
            res.append(l[2] - target_z)
            return res
        sol = root(residuals, self.init_guess, method='lm')
        if not sol.success: return None
        u, l, t = sol.x[0:3], sol.x[3:6], sol.x[6:9]
        spindle = self.spindle_tracker.get_new_position(u, l, t)
        spindle_end = self.axis_tracker.get_new_position(u, l, t)
        pr_mount = self.pr_tracker.get_new_position(u, l, t)
        shock_len, act_pts = self.actuator.solve(pr_mount)
        return {
            'upper_ball_joint': u, 'lower_ball_joint': l, 'tie_rod_upright': t, 
            'wheel_center': spindle, 'spindle_end': spindle_end, 'pushrod_upright_mount': pr_mount,
            'shock_len': shock_len, 'act_pts': act_pts,
            'upper_wishbone_front': p['upper_wishbone_front'], 'upper_wishbone_rear': p['upper_wishbone_rear'],
            'lower_wishbone_front': p['lower_wishbone_front'], 'lower_wishbone_rear': p['lower_wishbone_rear'],
            'tie_rod_chassis': tri_current, 'rocker_pivot_point': p['rocker_pivot_point'], 
            'shock_chassis_mount': p['shock_chassis_mount'], 'rocker_axis_definition': p['rocker_axis_definition']
        }
    def calculate_camber(self, res):
        kp = res['upper_ball_joint'] - res['lower_ball_joint']
        return np.degrees(np.arctan2(kp[1], kp[2]))
    def calculate_toe(self, res):
        vec = res['spindle_end'] - res['wheel_center']
        return np.degrees(np.arctan2(vec[0], vec[1]))

class AnalysisTools:
    def __init__(self, hp):
        self.hp = hp
    def get_instant_center_side(self):
        idx = [0, 2] # X, Z plane for Anti-Dive
        u1 = self.hp['upper_wishbone_front'][idx]; u2 = self.hp['upper_wishbone_rear'][idx]
        l1 = self.hp['lower_wishbone_front'][idx]; l2 = self.hp['lower_wishbone_rear'][idx]
        def intersect(p1, p2, p3, p4):
            x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
            denom = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1)
            if denom == 0: return None
            ua = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / denom
            return np.array([x1 + ua*(x2 - x1), y1 + ua*(y2 - y1)])
        return intersect(u1, u2, l1, l2)
    def get_anti_percentage(self, ic, cp_x, type='dive'):
        if ic is None: return 0.0
        dx = ic[0] - cp_x; dz = ic[1] - 0
        slope = dz / dx if dx != 0 else 0
        ratio = 1.0
        if type == 'dive': ratio = VEHICLE_PARAMS['brake_bias']
        pitch_ratio = VEHICLE_PARAMS['cg_height'] / VEHICLE_PARAMS['wheelbase']
        return (slope / pitch_ratio) * ratio * 100

class ForceSolver:
    def __init__(self, geo_res):
        self.geo = geo_res
        self.links = {
            'UCA_F': self.unit_vec(geo_res['upper_ball_joint'], geo_res['upper_wishbone_front']),
            'UCA_R': self.unit_vec(geo_res['upper_ball_joint'], geo_res['upper_wishbone_rear']),
            'LCA_F': self.unit_vec(geo_res['lower_ball_joint'], geo_res['lower_wishbone_front']),
            'LCA_R': self.unit_vec(geo_res['lower_ball_joint'], geo_res['lower_wishbone_rear']),
            'TieRod': self.unit_vec(geo_res['tie_rod_upright'], geo_res['tie_rod_chassis']),
            'Pushrod': self.unit_vec(geo_res['pushrod_upright_mount'], geo_res['act_pts']['pushrod_rocker_mount'])
        }
        self.cp = np.array([geo_res['wheel_center'][0], geo_res['wheel_center'][1], 0.0])
        self.arms = {
            'UCA_F': geo_res['upper_ball_joint']-self.cp, 'UCA_R': geo_res['upper_ball_joint']-self.cp,
            'LCA_F': geo_res['lower_ball_joint']-self.cp, 'LCA_R': geo_res['lower_ball_joint']-self.cp,
            'TieRod': geo_res['tie_rod_upright']-self.cp, 'Pushrod': geo_res['pushrod_upright_mount']-self.cp
        }
    def unit_vec(self, p1, p2):
        v = p2 - p1
        return v / np.linalg.norm(v)
    def solve(self, F_tire):
        A = np.zeros((6, 6))
        col = ['UCA_F', 'UCA_R', 'LCA_F', 'LCA_R', 'TieRod', 'Pushrod']
        for i, n in enumerate(col):
            u = self.links[n]; r = self.arms[n]; m = np.cross(r, u)
            A[0:3, i] = u; A[3:6, i] = m
        B = np.zeros(6); B[0:3] = -np.array(F_tire)
        try:
            x = np.linalg.solve(A, B)
            return dict(zip(col, x))
        except np.linalg.LinAlgError: return None

class LoadCaseGenerator:
    def __init__(self, params):
        self.p = params; self.g = 9.81
    def get_loads(self, g_long, g_lat):
        tw = self.p['mass_kg'] * self.g
        sf = tw * self.p['weight_dist']; sr = tw * (1 - self.p['weight_dist'])
        dl = (self.p['mass_kg'] * g_long * self.g * self.p['cg_height']) / self.p['wheelbase']
        dt = (self.p['mass_kg'] * g_lat * self.g * self.p['cg_height']) / self.p['track_width']
        fz_fl = (sf / 2) + (dl / 2) + (dt * self.p['weight_dist'])
        fx_fl = (self.p['mass_kg'] * g_long * self.g) * self.p['weight_dist'] / 2
        fy_fl = (self.p['mass_kg'] * g_lat * self.g) * self.p['weight_dist'] / 2
        fz_rl = (sr / 2) - (dl / 2) + (dt * (1 - self.p['weight_dist']))
        fx_rl = (self.p['mass_kg'] * g_long * self.g) * (1 - self.p['weight_dist']) / 2
        fy_rl = (self.p['mass_kg'] * g_lat * self.g) * (1 - self.p['weight_dist']) / 2
        return {'Front': [-fx_fl, fy_fl, fz_fl], 'Rear': [-fx_rl, fy_rl, fz_rl]}


# ============ PLOTTING ============

def plot_schematic_2d(ax, hp, title, view='front'):
    if view == 'front': 
        h, v, h_lab, v_lab = 1, 2, "Y (Lateral)", "Z (Vertical)"
        
        # Plot arms
        uca_f = hp['upper_wishbone_front'][[h,v]]; ubj = hp['upper_ball_joint'][[h,v]]
        lca_f = hp['lower_wishbone_front'][[h,v]]; lbj = hp['lower_ball_joint'][[h,v]]
        ax.plot([uca_f[0], ubj[0]], [uca_f[1], ubj[1]], 'g-', linewidth=2, label='Upper Arm')
        ax.plot([lca_f[0], lbj[0]], [lca_f[1], lbj[1]], 'b-', linewidth=2, label='Lower Arm')
        ax.plot([ubj[0], lbj[0]], [ubj[1], lbj[1]], 'r-', linewidth=3)
        
        # Plot actuation
        piv = hp['rocker_pivot_point'][[h,v]]; sh_out = hp['shock_rocker_mount'][[h,v]]
        sh_chas = hp['shock_chassis_mount'][[h,v]]; pr_in = hp['pushrod_rocker_mount'][[h,v]]
        pr_mnt = hp['pushrod_upright_mount'][[h,v]]
        ax.plot([piv[0], sh_out[0]], [piv[1], sh_out[1]], 'k-', linewidth=3)
        ax.plot([piv[0], pr_in[0]], [piv[1], pr_in[1]], 'k-', linewidth=3)
        ax.plot([sh_out[0], sh_chas[0]], [sh_out[1], sh_chas[1]], 'orange', linewidth=2, label='Shock')
        ax.plot([pr_in[0], pr_mnt[0]], [pr_in[1], pr_mnt[1]], 'm-', linewidth=1.5, label='Pushrod')
    else: 
        # Side view
        h, v, h_lab, v_lab = 0, 2, "X (Longitudinal)", "Z (Vertical)"
        
        # Pivot axes
        u1 = hp['upper_wishbone_front'][[h,v]]; u2 = hp['upper_wishbone_rear'][[h,v]]
        l1 = hp['lower_wishbone_front'][[h,v]]; l2 = hp['lower_wishbone_rear'][[h,v]]
        
        # UCA axis
        ax.plot([u1[0], u2[0]], [u1[1], u2[1]], 'g-o', linewidth=2, label='UCA Axis')
        # Extension
        if (u1[0]-u2[0]) != 0:
            m_u = (u1[1]-u2[1])/(u1[0]-u2[0]); b_u = u1[1] - m_u*u1[0]
            x_range = np.linspace(-2000, 3000, 2)
            ax.plot(x_range, m_u*x_range + b_u, 'g:', alpha=0.5)
            
        # LCA axis
        ax.plot([l1[0], l2[0]], [l1[1], l2[1]], 'b-o', linewidth=2, label='LCA Axis')
        if (l1[0]-l2[0]) != 0:
            m_l = (l1[1]-l2[1])/(l1[0]-l2[0]); b_l = l1[1] - m_l*l1[0]
            x_range = np.linspace(-2000, 3000, 2)
            ax.plot(x_range, m_l*x_range + b_l, 'b:', alpha=0.5)

    ax.set_title(title); ax.set_xlabel(h_lab); ax.set_ylabel(v_lab)
    ax.axis('equal'); ax.grid(True, linestyle=':', alpha=0.6)

def plot_line(ax, p1, p2, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], **kwargs)
def mirror_data(res):
    new = {}
    for k, v in res.items():
        if isinstance(v, np.ndarray): new[k] = np.array([v[0], -v[1], v[2]])
        elif k == 'act_pts' and v: new[k] = {ak: np.array([av[0], -av[1], av[2]]) for ak, av in v.items()}
        else: new[k] = v
    return new
def plot_wheel(ax, center, radius, width, camber, toe, color='k'):
    theta = np.linspace(0, 2*np.pi, 24)
    x = radius * np.cos(theta); z = radius * np.sin(theta); y = np.zeros_like(x)
    c_rad = np.radians(camber); t_rad = np.radians(toe)
    if abs(toe) > 135: t_rad -= np.pi 
    elif abs(toe) > 45: t_rad -= np.pi/2 * np.sign(toe) 
    Rx = np.array([[1, 0, 0], [0, np.cos(c_rad), -np.sin(c_rad)], [0, np.sin(c_rad), np.cos(c_rad)]])
    Rz = np.array([[np.cos(t_rad), -np.sin(t_rad), 0], [np.sin(t_rad), np.cos(t_rad), 0], [0, 0, 1]])
    def transform(y_off):
        pts = []; 
        for i in range(len(x)): 
            p = np.array([x[i], y[i] + y_off, z[i]]); pts.append((Rz @ (Rx @ p)) + center)
        return np.array(pts)
    outer = transform(width/2); inner = transform(-width/2)
    ax.plot(outer[:,0], outer[:,1], outer[:,2], color=color, linewidth=2)
    ax.plot(inner[:,0], inner[:,1], inner[:,2], color=color, linewidth=2)
    for i in range(len(theta)):
        ax.plot([inner[i,0], outer[i,0]], [inner[i,1], outer[i,1]], [inner[i,2], outer[i,2]], color=color, alpha=0.3)
def plot_corner(ax, res, c, tire_params=None):
    plot_line(ax, res['upper_wishbone_front'], res['upper_wishbone_rear'], color='grey', linestyle='--', alpha=0.5)
    plot_line(ax, res['lower_wishbone_front'], res['lower_wishbone_rear'], color='grey', linestyle='--', alpha=0.5)
    plot_line(ax, res['upper_wishbone_front'], res['upper_ball_joint'], color=c, linewidth=2)
    plot_line(ax, res['upper_wishbone_rear'], res['upper_ball_joint'], color=c, linewidth=2)
    plot_line(ax, res['lower_wishbone_front'], res['lower_ball_joint'], color=c, linewidth=2)
    plot_line(ax, res['lower_wishbone_rear'], res['lower_ball_joint'], color=c, linewidth=2)
    plot_line(ax, res['tie_rod_chassis'], res['tie_rod_upright'], color='c', linewidth=2)
    plot_line(ax, res['upper_ball_joint'], res['lower_ball_joint'], color='k', linewidth=2)
    plot_line(ax, res['upper_ball_joint'], res['tie_rod_upright'], color='k', linewidth=1)
    plot_line(ax, res['lower_ball_joint'], res['tie_rod_upright'], color='k', linewidth=1)
    plot_line(ax, res['lower_ball_joint'], res['wheel_center'], color='k', linewidth=3)
    if tire_params:
        kp = res['upper_ball_joint'] - res['lower_ball_joint']; camber = np.degrees(np.arctan2(kp[1], kp[2]))
        vec = res['spindle_end'] - res['wheel_center']; toe = np.degrees(np.arctan2(vec[0], vec[1]))
        plot_wheel(ax, res['wheel_center'], tire_params['tire_radius'], tire_params['tire_width'], camber, toe)
    if res['act_pts']:
        plot_line(ax, res['pushrod_upright_mount'], res['act_pts']['pushrod_rocker_mount'], color='m', linewidth=1.5)
        plot_line(ax, res['rocker_pivot_point'], res['act_pts']['pushrod_rocker_mount'], color='g')
        plot_line(ax, res['rocker_pivot_point'], res['act_pts']['shock_rocker_mount'], color='g')
        plot_line(ax, res['act_pts']['shock_rocker_mount'], res['shock_chassis_mount'], color='orange', linewidth=3)
        p1 = res['rocker_pivot_point']; p2 = res['rocker_axis_definition']
        vec = p2 - p1; start = p1 - vec*0.5; end = p1 + vec*1.5
        plot_line(ax, start, end, color='y', linestyle=':', linewidth=2, alpha=0.7)
def set_axes_proportional(ax):
    lims = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    ranges = [abs(l[1]-l[0]) for l in lims]; mids = [np.mean(l) for l in lims]
    r = 0.5 * max(ranges)
    ax.set_xlim3d([mids[0]-r, mids[0]+r]); ax.set_ylim3d([mids[1]-r, mids[1]+r]); ax.set_zlim3d([mids[2]-r, mids[2]+r])
    ax.set_box_aspect((1, 1, 1))
def plot_rocker_2d(ax, res, title):
    pass # using plot_schematic_2d instead