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
