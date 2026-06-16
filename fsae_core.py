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
        res = self.hp.copy()

        offset = np.array([0, 0, h])

        # Move upright-related points
        res['upper_ball_joint'] = self.hp['upper_ball_joint'] + offset
        res['lower_ball_joint'] = self.hp['lower_ball_joint'] + offset
        res['tie_rod_upright'] = self.hp['tie_rod_upright'] + offset
        res['wheel_center'] = self.hp['wheel_center'] + offset
        res['pushrod_upright_mount'] = self.hp['pushrod_upright_mount'] + offset

        # ✅ CRITICAL: fake spindle end (needed for toe calc in plot)
        res['spindle_end'] = res['wheel_center'] + np.array([50, 0, 0])

        # ✅ CRITICAL: actuation points (THIS is why shocks disappeared)
        res['act_pts'] = {
            'pushrod_rocker_mount': self.hp['pushrod_rocker_mount'],
            'shock_rocker_mount': self.hp['shock_rocker_mount']
        }

        # Shock length (placeholder but needed)
        res['shock_len'] = np.linalg.norm(
            self.hp['shock_rocker_mount'] - self.hp['shock_chassis_mount']
        )

        return res

    def calculate_camber(self, r):
        vec = r['upper_ball_joint'] - r['lower_ball_joint']
        return np.degrees(np.arctan2(vec[1], vec[2]))

    def calculate_toe(self, r):
        vec = r['spindle_end'] - r['wheel_center']
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
        return np.linalg.lstsq(np.eye(3), -np.array(F), rcond=None)[0]


# ================= FULL VISUALIZATION =================

def mirror_data(res):
    new = {}
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            new[k] = np.array([v[0], -v[1], v[2]])
        else:
            new[k] = v
    return new


def plot_wheel(ax, center, radius, width, camber=0, toe=0, color='k'):
    theta = np.linspace(0, 2*np.pi, 30)

    # wheel circle (local)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    y = np.zeros_like(x)

    # rotations
    cam = np.radians(camber)
    toe = np.radians(toe)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(cam), -np.sin(cam)],
        [0, np.sin(cam), np.cos(cam)]
    ])

    Rz = np.array([
        [np.cos(toe), -np.sin(toe), 0],
        [np.sin(toe),  np.cos(toe), 0],
        [0, 0, 1]
    ])

    def transform(y_offset):
        pts = []
        for i in range(len(x)):
            p = np.array([x[i], y[i] + y_offset, z[i]])
            p = Rz @ (Rx @ p)
            pts.append(p + center)
        return np.array(pts)

    outer = transform(width/2)
    inner = transform(-width/2)

    ax.plot(outer[:,0], outer[:,1], outer[:,2], color=color, linewidth=2)
    ax.plot(inner[:,0], inner[:,1], inner[:,2], color=color, linewidth=2)

    # spokes
    for i in range(len(theta)):
        ax.plot(
            [inner[i,0], outer[i,0]],
            [inner[i,1], outer[i,1]],
            [inner[i,2], outer[i,2]],
            color=color, alpha=0.3
        )


def plot_corner(ax, res, c, params=None):

    def line(p1, p2, **kwargs):
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            **kwargs
        )

    # ---------------- WISHBONES ----------------
    line(res['upper_wishbone_front'], res['upper_wishbone_rear'], 
         color='grey', linestyle='--', alpha=0.5)

    line(res['lower_wishbone_front'], res['lower_wishbone_rear'], 
         color='grey', linestyle='--', alpha=0.5)

    line(res['upper_wishbone_front'], res['upper_ball_joint'], color=c, linewidth=2)
    line(res['upper_wishbone_rear'],  res['upper_ball_joint'], color=c, linewidth=2)

    line(res['lower_wishbone_front'], res['lower_ball_joint'], color=c, linewidth=2)
    line(res['lower_wishbone_rear'],  res['lower_ball_joint'], color=c, linewidth=2)

    # ---------------- UPRIGHT ----------------
    line(res['upper_ball_joint'], res['lower_ball_joint'], color='k', linewidth=2)

    line(res['upper_ball_joint'], res['tie_rod_upright'], color='k', linewidth=1)
    line(res['lower_ball_joint'], res['tie_rod_upright'], color='k', linewidth=1)

    # ---------------- TIE ROD ----------------
    line(res['tie_rod_chassis'], res['tie_rod_upright'], color='c', linewidth=2)

    # ---------------- WHEEL LINK ----------------
    line(res['lower_ball_joint'], res['wheel_center'], color='k', linewidth=3)

    # ---------------- PUSHROD ----------------
    if 'pushrod_upright_mount' in res:
        line(res['pushrod_upright_mount'],
             res['rocker_pivot_point'],
             color='m', linewidth=2)

    # ---------------- ROCKER + SHOCK ----------------
    if 'rocker_pivot_point' in res:
        line(res['rocker_pivot_point'],
             res['pushrod_rocker_mount'],
             color='g')

        line(res['rocker_pivot_point'],
             res['shock_rocker_mount'],
             color='g')

        line(res['shock_rocker_mount'],
             res['shock_chassis_mount'],
             color='orange', linewidth=3)

        # rocker axis
        p1 = res['rocker_pivot_point']
        p2 = res['rocker_axis_definition']
        vec = p2 - p1
        start = p1 - vec * 0.5
        end = p1 + vec * 1.5

        line(start, end, color='y', linestyle=':', linewidth=2)

    # ---------------- WHEEL ----------------
    if params:
        kp = res['upper_ball_joint'] - res['lower_ball_joint']
        camber = np.degrees(np.arctan2(kp[1], kp[2]))

        vec = res['tie_rod_upright'] - res['wheel_center']
        toe = np.degrees(np.arctan2(vec[0], vec[1]))

        plot_wheel(
            ax,
            res['wheel_center'],
            params['tire_radius'],
            params['tire_width'],
            camber,
            toe,
            color=c
        )


def set_axes_proportional(ax):
    lims = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    ranges = [abs(l[1] - l[0]) for l in lims]
    mids = [np.mean(l) for l in lims]

    r = 0.5 * max(ranges)

    ax.set_xlim3d([mids[0] - r, mids[0] + r])
    ax.set_ylim3d([mids[1] - r, mids[1] + r])
    ax.set_zlim3d([mids[2] - r, mids[2] + r])
    ax.set_box_aspect((1, 1, 1))
