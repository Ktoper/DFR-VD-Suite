import numpy as np
from scipy.optimize import root
import pandas as pd

# ================= VEHICLE PARAMS =================
VEHICLE_PARAMS = {
    'mass_kg': 320.0,
    'cg_height': 280.0,
    'wheelbase': 1650.0,
    'track_width': 1270.0,
    'weight_dist': 0.50,
    'tire_radius': 228.6,
    'tire_width': 205.0,
    'brake_bias': 0.60,
    'unsprung_mass_corner': 15.0,
    'target_freq_front': 3.0,
    'target_freq_rear': 3.5,
}

# ================= HARDPOINTS =================

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
    'pushrod_upright_mount': np.array([302.292, 562.587, 164.418]),
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
    'pushrod_upright_mount': np.array([-1348.963, 552.751, 168.683]),
    'rocker_pivot_point':     np.array([-1426.258, 260.411, 336.729]),
    'rocker_axis_definition': np.array([-1522.266, 270.886, 310.792]),
    'pushrod_rocker_mount':   np.array([-1428.888, 288.292, 357.722]),
    'shock_rocker_mount':     np.array([-1445.120, 218.736, 389.714]),
    'shock_chassis_mount':    np.array([-1451.099, 50.000, 343.698]),
}

# ================= HELPERS =================
def dict_to_df(hp):
    data = []
    for k, v in hp.items():
        data.append({'Point': k, 'X': v[0], 'Y': v[1], 'Z': v[2]})
    return pd.DataFrame(data).set_index('Point')

def df_to_dict(df):
    hp_new = {}
    for idx, row in df.iterrows():
        hp_new[idx] = np.array([row['X'], row['Y'], row['Z']])
    return hp_new

def calculate_damping_ratios(df, mass, spring, mr):
    k_wheel = spring * (mr**2) * 1000
    c_crit = 2 * np.sqrt(k_wheel * mass)

    res = df.copy()
    res['Cs'] = res['force_n'] / res['velocity_ms'].replace(0, 1e-9)
    res['Cw'] = res['Cs'] * (mr**2)
    res['Zeta'] = res['Cw'] / c_crit
    return res, c_crit

# ================= SOLVERS =================
class SuspensionSolver:
    def __init__(self, hp):
        self.hp = hp
        self.init_guess = None

    def solve_heave(self, h):
        # Simplified placeholder (your real solver logic goes here)
        res = self.hp.copy()
        res['wheel_center'] = self.hp['wheel_center'] + np.array([0, 0, h])

        res['shock_len'] = 200 + h * 0.5
        res['act_pts'] = None
        return res

    def calculate_camber(self, r):
        vec = r['upper_ball_joint'] - r['lower_ball_joint']
        return np.degrees(np.arctan2(vec[1], vec[2]))

    def calculate_toe(self, r):
        vec = r['tie_rod_upright'] - r['wheel_center']
        return np.degrees(np.arctan2(vec[0], vec[1]))

# ================= LOADS =================
class LoadCaseGenerator:
    def __init__(self, p):
        self.p = p
        self.g = 9.81

    def get_loads(self, g_long, g_lat):
        m = self.p['mass_kg']
        wd = self.p['weight_dist']

        fx = m * g_long * self.g / 2
        fy = m * g_lat * self.g / 2
        fz_f = m * self.g * wd / 2
        fz_r = m * self.g * (1 - wd) / 2

        return {
            'Front': [-fx, fy, fz_f],
            'Rear':  [-fx, fy, fz_r]
        }

# ================= FORCES =================
class ForceSolver:
    def __init__(self, geo):
        self.geo = geo

    def solve(self, F):
        # Dummy stable solve
        A = np.eye(3)
        return np.linalg.lstsq(A, -np.array(F), rcond=None)[0]
