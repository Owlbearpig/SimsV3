from numpy import pi
from numpy import array
from pathlib import Path
import os
from scipy.constants import c as c0

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

rad = 180 / pi
THz = 10**12
GHz = 10**9
m_um = 10**6 # m to um conversion
um = 10**-6

plot_data_dir = Path('plot_data')
