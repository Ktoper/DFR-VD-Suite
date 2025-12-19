DFR - Vehicle Dynamics Suite

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dartmouth-vd-suite.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This **Vehicle Dynamics Suite** is a custom-developed engineering tool designed as a proof-of-concept for the **Dartmouth Formula Racing (DFR)** team to optimize suspension geometry and analyze vehicle behavior for the 2026 competition season.

Built with **Python** and **Streamlit**, this suite is a test for replacing legacy software and spreadsheets with a physics-based solver capable of simple 3D kinematic simulation, force analysis, and damping optimization.

### Key Capabilities
*3D Kinematics Engine**Solves spatial linkages for Double Wishbone pushrod suspension.
**Bump & Roll Analysis:** visualizes Camber Gain, Bump Steer, and Motion Ratios.
**Steering Geometry:** Analyzes Ackermann percentage and steering errors.
**Damping Calculator:** Ingests shock dyno data (.csv) to calculate Damping Ratios ($\zeta$) for critical damping analysis.
**Load Case Generation:** Automates FEA load extraction for braking, cornering, and acceleration events.


## Structure
* `app.py`: The main user interface and visualization dashboard.
* `fsae_core.py`: The physics engine containing the rigid body solvers (`SuspensionSolver`) and math logic.
* `requirements.txt`: List of Python dependencies.

## Guide
1.  **Geometry Editor:** Input your hardpoint coordinates (X, Y, Z) for the Front and Rear corners.
2.  **Kinematics Tab:** Visualize how suspension parameters change through Bump/Droop travel.
3.  **Damping Tab:** Upload a `.csv` from your shock dyno to check low/high-speed damping ratios against the FSAE target ($\zeta \approx 0.65-0.75$).
4.  **FEA Loads:** Generate reaction forces for your wishbones and pushrods to use in ANSYS/SolidWorks Simulation.

## License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Developed by KT - DFR VD Lead*
