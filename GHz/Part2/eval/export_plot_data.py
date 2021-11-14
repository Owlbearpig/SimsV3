from generate_plotdata import export_csv
import numpy as np
from consts import *
from simulation import sim_polarization_parameters
from results import p2_angles, p2_d

polOffset = 1.25

f_meas, delta_meas, rel_meas = np.load('f_meas.npy'), np.load(f'delta_meas_{polOffset}degPolOffset.npy'), \
                               np.load(f'rel_meas_{polOffset}degPolOffset.npy')

range_mask_meas = (75*GHz < f_meas)*(f_meas < 110*GHz)

data_meas = {'freq': f_meas[range_mask_meas],
             'delta': delta_meas[range_mask_meas],
             'rel': rel_meas[range_mask_meas]
}

export_csv(data_meas, path=f'meas_result_{polOffset}degPolOffset.csv')

p2_x = np.concatenate((p2_angles, p2_d))

f_sim, parameters = sim_polarization_parameters(p2_x, rez=1)
a, b = parameters.ellipse_axes()

range_mask_sim = (75*GHz < f_sim)*(f_sim < 110*GHz)

data_sim = {'freq': f_sim[range_mask_sim],
            'delta': parameters.delay()[range_mask_sim],
            'rel': (b / a)[range_mask_sim],
}

#export_csv(data_sim, path='simulation_0degPolOffset.csv')
