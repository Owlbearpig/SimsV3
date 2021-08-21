from numpy import pi
from pathlib import Path
import os

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

rad = 180 / pi
THz = 10**12
m_um = 10**6 # m to um conversion

plot_data_dir = Path('plot_data')
