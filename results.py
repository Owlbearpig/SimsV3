import numpy as np
from numpy import array, flip

angles_cl4_05_22_n6 = array([6.45706370e+00, 3.23172880e+00, 1.62683796e+00, 3.38754487e+00,1.53410411e+00, 3.53553954e+00])
d_cl4_05_22_n6 = array([5.25952607e+03, 1.99339481e+03,8.33507973e+03, 4.27282730e+03, 4.25283016e+03, 2.12689058e+03])
x_cl4_05_22_n6 = np.concatenate((angles_cl4_05_22_n6, d_cl4_05_22_n6))

angles_cl4_04_24_n6 = array([6.15370628e-01, 1.30340254e+00, 2.90403823e+00, 3.45621387e-01, 2.24195442e+00, 2.35872063e-01])
d_cl4_04_24_n6 = array([1.06871635e+03, 8.34445653e+03, 4.11903805e+03, 2.10233555e+03, 2.09616433e+03, 6.35692990e+03])
x_cl4_04_24_n6 = np.concatenate((angles_cl4_04_24_n6, d_cl4_04_24_n6))

angles_cl4_05_20_n6 = array([6.28519254e+00, 4.64743972e+00, 6.81758339e+00, 2.29016774e+00, 6.66441738e+00, 1.17972161e+00])
d_cl4_05_20_n6 = array([5.95435042e+03, 9.51799993e+03, 4.79662256e+03, 4.77927465e+03, 4.75038499e+03, 2.40089682e+03])
x_cl4_05_20_n6 = np.concatenate((angles_cl4_05_20_n6, d_cl4_05_20_n6))

angles_cl4_02_20_n6 = array([5.33985248e+00,  9.56709262e-01, -4.97426771e-01, -1.46739563e-02, 5.59238259e-01,  7.63377294e-02])
d_cl4_02_20_n6 = array([2.73466001e+03,  8.24960819e+03, 2.00002995e+03,  1.91378856e+03,  2.72786597e+03,  5.47221231e+03])
x_cl4_02_20_n6 = np.concatenate((angles_cl4_02_20_n6, d_cl4_02_20_n6))

angles_cl4_02_15_n6 = array([8.97059689e-01, 3.39617650e+00, 1.24729401e+00, 2.75500826e+00, 5.59629715e+00, 4.46653490e+00])
d_cl4_02_15_n6 = array([1.68288798e+03, 6.67721916e+03, 6.77502950e+03, 3.43409664e+03, 3.34206925e+03, 1.00276008e+04])
x_cl4_02_15_n6 = np.concatenate((angles_cl4_02_15_n6, d_cl4_02_15_n6))

angles_cl4_02_20 = array([4.61953041e+00, 4.58461298e-01, 2.48041345e+00, 3.78437858e+00, 5.25502234e+00])
d_cl4_02_20 = array([3.91043128e+03, 2.60763905e+03, 2.59679197e+03, 1.04906116e+04, 5.27866155e+03]) # 24.9
x_cl4_02_20 = np.concatenate((angles_cl4_02_20, d_cl4_02_20))

angles_cl4_035_20 = array([1.55065274e+00,  3.02220562e-01, -3.14806903e-01,  1.34268867e+00, 1.73947220e-01])
d_cl4_035_20 = array([3.90855784e+03,  5.21546932e+03,  5.06268420e+03, 7.66853582e+03,  7.75647746e+03]) # 29.6
x_cl4_035_20 = np.concatenate((angles_cl4_035_20, d_cl4_035_20))

angles_cl4_05_20 = array([4.70121322e+00,  6.53286936e+00,  4.50875188e+00,  1.76718691e+00, 3.21159687e+00,])
d_cl4_05_20 = array([3.45388897e+03,  4.57809315e+03,  2.29421050e+03, -2.38574802e+03,  4.48955958e+03]) # 17.3
x_cl4_05_20 = np.concatenate((angles_cl4_05_20, d_cl4_05_20))

angles_cl4_05_15 = array([3.53388516e+00, 1.90528314e+00, 3.70143600e+00, 1.86344386e+00, 4.24715276e+00])
d_cl4_05_15 = array([7.71441240e+03, 6.13007734e+03, 3.05190115e+03, 3.08095056e+03, 3.10321319e+03])
x_cl4_05_15 = np.concatenate((angles_cl4_05_15, d_cl4_05_15))

angles_m = flip(np.deg2rad(array([31.7, 10.4, 118.7, 24.9, 5.1, 69.0])), 0)
d_m = flip(array([3360, 6730, 6460, 3140, 3330, 8430]), 0)
x_ml4 = np.concatenate((angles_m, d_m))

# obtained using wrong erf; (Int_x - Int_y)**2 # Although it's result that got printed ...
"""
angles_cl4 = np.deg2rad(array([3.12, 112.71, 144.85, 83.07, 97.93]))
d_cl4 = array([2438.4, 3088.1, 1683.1, 1454.2, 2718.4])
x_ceramic_l4 = np.concatenate((angles_cl4, d_cl4))
"""

########################################################################################################################
# example:
"""
result1 = {
        'name': 'c_l4_02_15', # name for plotdata dir
        'comments': '', # anything
        'x': x_cl4_02_15_n6, # concatenate((angles, d)) or concatenate((angles, d, stripes)) 
        'bf': 'intrinsic', # or 'form'
        'mat_name': ('fast_mat_name', 'slow_mat_name') # if form, only 'fast_mat_name' is used. 'slow_mat_name' is air
}
"""

result1 = {
        'name': 'c_l4_02_15',
        'comments': '',
        'x': x_cl4_02_15_n6,
        'bf': 'intrinsic',
        'mat_name': ('ceramic_fast', 'ceramic_slow')
}

result_masson = {
        'name': 'masson',
        'comments': '',
        'x': x_ml4,
        'bf': 'intrinsic',
        'mat_name': ('quartz_sellmeier_fast', 'quartz_sellmeier_slow')
}