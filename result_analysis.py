import numpy as np
from numpy import (power, outer, sqrt, exp, sin, cos, conj, dot, pi,
                   einsum, arctan, array, arccos, conjugate, flip, angle)
import pandas
from pathlib import Path, PureWindowsPath
import scipy
from scipy.constants import c as c0
from scipy.optimize import basinhopping
from py_pol import jones_matrix
from py_pol import jones_vector
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from numpy.linalg import solve
import string
import matplotlib.pyplot as plt
import sys

THz = 10**12
m_um = 10**6 # m to um conversion

def load_material_data(mat_name):

    mat_paths = {
        'ceramic_slow': Path('material_data/Sample1_000deg_1825ps_0m-2Grad_D=3000.csv'),
        'ceramic_fast': Path('material_data/Sample1_090deg_1825ps_0m88Grad_D=3000.csv'),
        'HIPS_MUT_1_1': Path('material_data/MUT 1-1.csv'),
        'Fused_4eck': Path('material_data/4Eck_D=2042.csv'),
        'quartz_m_slow': Path('material_data/quartz_m_slow.csv'),
        'quartz_m_fast': Path('material_data/quartz_m_fast.csv'),
        'quartz_m_slow2': Path('material_data/quartz_m_slow2.csv'),
        'quartz_m_fast2': Path('material_data/quartz_m_fast2.csv'),
    }

    df = pandas.read_csv(mat_paths[mat_name])

    freq_dict_key = [key for key in df.keys() if "freq" in key][0]
    eps_mat_r_key = [key for key in df.keys() if "epsilon_r" in key][0]
    eps_mat_i_key = [key for key in df.keys() if "epsilon_i" in key][0]

    frequencies = np.array(df[freq_dict_key])

    data_slice = np.where((frequencies > f_min) &
                          (frequencies < f_max))
    data_slice = data_slice[0][::resolution]

    eps_mat_r = np.array(df[eps_mat_r_key])[data_slice]
    eps_mat_i = np.array(df[eps_mat_i_key])[data_slice]

    eps_mat1 = (eps_mat_r + eps_mat_i * 1j).reshape(len(data_slice), 1)

    return eps_mat1, frequencies[data_slice].reshape(len(data_slice), 1)


def form_birefringence(stripes):
    """
    :return: array with length of frequency, frequency resolved [ns, np, ks, kp]
    """

    l_mat1, l_mat2 = stripes

    a = (1 / 3) * power(outer(1 / wls, (l_mat1 * l_mat2 * pi) / (l_mat1 + l_mat2)), 2)

    # first order s and p
    wp_eps_s_1 = outer((eps_mat2 * eps_mat1), (l_mat2 + l_mat1)) / (
            outer(eps_mat2, l_mat1) + outer(eps_mat1, l_mat2))

    wp_eps_p_1 = outer(eps_mat1, l_mat1 / (l_mat2 + l_mat1)) + outer(eps_mat2, l_mat2 / (l_mat2 + l_mat1))

    # 2nd order
    wp_eps_s_2 = wp_eps_s_1 + (a * power(wp_eps_s_1, 3) * wp_eps_p_1 * power((1 / eps_mat1 - 1 / eps_mat2), 2))
    wp_eps_p_2 = wp_eps_p_1 + (a * power((eps_mat1 - eps_mat2), 2))

    # returns
    n_p, n_s = (
        sqrt(abs(wp_eps_p_2) + wp_eps_p_2.real) / sqrt(2),
        sqrt(abs(wp_eps_s_2) + wp_eps_s_2.real) / sqrt(2)
    )
    k_p, k_s = (
        sqrt(abs(wp_eps_p_2) - wp_eps_p_2.real) / sqrt(2),
        sqrt(abs(wp_eps_s_2) - wp_eps_s_2.real) / sqrt(2)
    )

    return np.array([n_s, n_p, k_s, k_p])


def opt(n, x=None, ret_j=False):
    #d = np.ones(n)*204#*4080/n
    #d = flip(np.array([3360, 6730, 6460, 3140, 3330, 8430]), 0)

    # setup einsum_str
    s0 = string.ascii_lowercase + string.ascii_uppercase

    einsum_str = ''
    for i in range(n):
        einsum_str += s0[n + 2] + s0[i] + s0[i + 1] + ','

    # remove last comma
    einsum_str = einsum_str[:-1]
    einsum_str += '->' + s0[n + 2] + s0[0] + s0[n]

    # einsum path
    test_array = np.zeros((n, m, 2, 2))
    einsum_path = np.einsum_path(einsum_str, *test_array, optimize='greedy')

    # calc matrix chain from m_n_2_2 tensor
    # (If n > 2x alphabet length, einsum breaks -> split into k parts... n/k)
    """
    1_n_(2x2)*...*1_3_(2x2)*1_2_(2x2)*1_1_(2x2) -> (2x2)_1
    2_n_(2x2)*...*2_3_(2x2)*2_2_(2x2)*2_1_(2x2) -> (2x2)_2
    .
    .
    m_n_(2x2)*...*m_3_(2x2)*m_2_(2x2)*m_1_(2x2) -> (2x2)_m
    """
    def matrix_chain_calc(matrix_array):
        return


    def erf(x):
        #d, angles = x[0:n], x[n:2*n]
        #d, angles = x[0:n], np.deg2rad(x[n:2*n])
        #angles = np.deg2rad(x[0:n])

        j = np.zeros((m, n, 2, 2), dtype=complex)

        angles, d = x[0:n], x[n:2*n]

        phi_s, phi_p = (2 * n_s * pi / wls) * d.T, (2 * n_p * pi / wls) * d.T
        alpha_s, alpha_p = -(2 * pi * k_s / wls) * d.T, -(2 * pi * k_p / wls) * d.T
        #alpha_s, alpha_p = np.zeros_like(wls), -(2 * pi * (k_p - k_s) / wls) * d.T

        x, y = 1j * phi_s + alpha_s, 1j * phi_p + alpha_p
        angles = np.tile(angles, (m, 1))

        j[:, :, 0, 0] = exp(y) * sin(angles) ** 2 + exp(x) * cos(angles) ** 2
        j[:, :, 0, 1] = 0.5 * sin(2 * angles) * (exp(x)-exp(y))
        j[:, :, 1, 0] = j[:, :, 0, 1]
        j[:, :, 1, 1] = exp(x) * sin(angles) ** 2 + exp(y) * cos(angles) ** 2

        np.einsum(einsum_str, *j.transpose((1, 0, 2, 3)), out=j[:, 0], optimize=einsum_path[0])

        j = j[:, 0]

        # TODO add optimization(fix bounds especially thickness bounds)
        if ret_j:
            return j

        #delta_equiv = 2*arccos(0.5*np.abs(j[:, 0, 0]+conjugate(j[:, 0, 0])))

        # hwp 1 int opt
        #res = (1 / m) * sum((1 - j[:, 1, 0] * conj(j[:, 1, 0])) ** 2 + (j[:, 0, 0] * conj(j[:, 0, 0])) ** 2)

        # hwp 2 int opt
        #res = (1 / m) * sum((1 - j[:, 1, 0].real)**2 + (j[:, 1, 0].imag) ** 2 + (j[:, 0, 0] * conj(j[:, 0, 0])) ** 2)

        # hwp 3 mat opt
        #print(((np.angle(j[:,1,0])-np.angle(j[:,0,1]))**2))
        #print((j[:, 1, 0].imag - j[:, 0, 1].imag) ** 2)
        #print()
        """
        res = (1 / m) * sum(np.absolute(j[:,0,0])**2+np.absolute(j[:,1,1])**2+
                            #(1-np.abs(j[:,1,0].real))**2+(1-np.abs(j[:,0,1].real))**2)
                            (1-j[:,1,0].real)+(1-j[:,0,1].real)+
                            (j[:,1,0].imag)**2+(j[:,0,1].imag)**2)
        """

        # hwp 4 mat opt back to start
        #res = sum(np.absolute(j[:, 0, 0]) ** 2 + np.absolute(j[:, 1, 1]) ** 2) \
        #+ sum((1-j[:, 0, 1].imag) ** 2 + (1-j[:, 1, 0].imag) ** 2)

        # qwp state opt
        q = j[:, 0, 0] / j[:, 1, 0]
        res = sum(q.real ** 2 + (q.imag - 1) ** 2)
        #res = sum((j[:, 1, 0] * conj(j[:, 1, 0]) - j[:, 0, 0] * conj(j[:, 0, 0])) ** 2)
        # qwp state opt 2.
        #a, b = j[:, 0, 0], j[:, 1, 0]
        #phi = angle(a)-angle(b)
        #res = sum((np.abs(b)-np.abs(a))**2+(phi-pi/2)**2)

        # Masson ret. opt.
        #A, B = j[:, 0, 0], j[:, 0, 1]
        #delta_equiv = 2 * arctan(sqrt((A.imag ** 2 + B.imag ** 2) / (A.real ** 2 + B.real ** 2)))
        #res = (1/m)*np.sum((delta_equiv-pi)**2)

        return res

    def print_fun(x, f, accepted):
        print(x, f, accepted)

    bounds = list(zip([0]*n, [2*pi]*n)) + list(zip([0]*n, [10**4]*n))

    minimizer_kwargs = {}

    class MyBounds(object):
        def __init__(self):
            self.xmax = np.ones(2*n)*(16*pi)
            self.xmin = np.ones(2*n)*(0)
        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))

            return (tmax and tmin)

    mybounds = MyBounds()

    """ Custom step-function """
    class RandomDisplacementBounds(object):
        """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
            Modified! (dropped acceptance-rejection sampling for a more specialized approach)
        """
        def __init__(self, xmin, xmax, stepsize=2.4):
            self.xmin = xmin
            self.xmax = xmax
            self.stepsize = stepsize

        def __call__(self, x):
            """take a random step but ensure the new position is within the bounds """
            min_step = np.maximum(self.xmin - x, -self.stepsize)
            max_step = np.minimum(self.xmax - x, self.stepsize)

            random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
            xnew = x + random_step

            return xnew

    bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))

    x0 = np.concatenate((np.random.random(n)*2*pi, np.random.random(n)*10**4))

    if ret_j:
        return erf(x)

    #return minimize(erf, x0)
    return basinhopping(erf, x0, niter=200, callback=print_fun, take_step=bounded_step, disp=True, T=25)


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

if __name__ == '__main__':
    from dataexport import save, pe_export
    #f = (np.arange(0.2, 2.0, 0.05)*THz)[:]
    resolution = 1
    f_min, f_max = 0.2*THz, 2.5*THz

    eps_mat1, f = load_material_data('ceramic_fast')
    eps_mat2, _ = load_material_data('ceramic_slow')
    #print(len(f))
    wls = (c0/f)*m_um
    m = len(wls)

    #stripes = 628, 517.1
    #n_s, n_p, k_s, k_p = form_birefringence(stripes)
    n_s, n_p = sqrt(np.abs(eps_mat1)+eps_mat1.real)/sqrt(2), sqrt(np.abs(eps_mat2)+eps_mat2.real)/sqrt(2)
    k_s, k_p = sqrt(np.abs(eps_mat1)-eps_mat1.real)/sqrt(2), sqrt(np.abs(eps_mat2)-eps_mat2.real)/sqrt(2)

    n = 6
    """
    best, best_res = np.inf, None
    for _ in range(10):
        ret = opt(n=n)
        print(ret)
        print(ret, _)
        if ret.fun < best:
            best = ret.fun
            best_res = ret

    print(best_res)
    """
    j = opt(n=n, ret_j=True, x=x_cl4_02_15_n6)

    #int_x = j[:, 0, 0]*np.conjugate(j[:, 0, 0])
    #int_y = j[:, 1, 0]*np.conjugate(j[:, 1, 0])
    q = j[:, 0, 0] / j[:, 1, 0]
    L = q.real ** 2 + (q.imag - 1) ** 2
    #L = L / max(L)

    #A, B = j[:, 0, 0], j[:, 0, 1]
    #delta_equiv = 2 * arctan(sqrt((A.imag ** 2 + B.imag ** 2) / (A.real ** 2 + B.real ** 2)))
    #L = delta_equiv/pi

    #save({'freq': f.flatten(), 'L': L}, name='x_cl4_05_22_n6')

    #plt.plot(f, L)
    #plt.show()

    #int_x, int_y = 10*np.log10(int_x.real), 10*np.log10(int_y.real)

    J = jones_matrix.create_Jones_matrices('cl4')
    J.from_matrix(j)

    #J.remove_global_phase()
    #J.set_global_phase(0)
    #J.analysis.retarder(verbose=True)
    #print(j[:,0,0])
    #print(j[:, 0, 1])
    #print(j[:, 1, 0])
    #print(j[:, 1, 1])
    #v1, v2, E1, E2 = J.parameters.eig(as_objects=True)
    #E1.draw_ellipse()
    #E2.draw_ellipse()
    #plt.show()
    #J.parameters.global_phase(verbose=True)
    #Jqwp = jones_matrix.create_Jones_matrices()
    #Jqwp.retarder_linear()

    #print(J.analysis.retarder(verbose=True))
    #E1[11].draw_ellipse()
    #E2[11].draw_ellipse()
    #plt.show()
    #E1.draw_ellipse()
    #E2.draw_ellipse()

    #print(360-np.abs(np.angle(v1[0])-np.angle(v2[0]))*180/pi)
    #E1.parameters.global_phase(verbose=True)
    #(J*E1).parameters.global_phase(verbose=True)
    #E2.remove_global_phase()
    #(J * E2).parameters.global_phase(verbose=True)
    #J.parameters.retardance(verbose=True)
    #print(np.angle(j)[:,0,0]-np.angle(j)[:,0,1])
    #v1, v2, E1, E2 = Jqwp.parameters.eig(as_objects=True)
    #E1.draw_ellipse()
    #E2.draw_ellipse()
    #plt.show()

    #Jhi.remove_global_phase()
    #v1, v2, E1, E2 = Jhi.parameters.eig(as_objects=True)
    #E1.draw_ellipse()
    #E2.draw_ellipse()
    #plt.show()
    #plt.show()
    #print(Jhi.parameters)
    #J.remove_global_phase()
    #print(J[11].parameters)
    #J_qwp = jones_matrix.create_Jones_matrices('J_qwp')
    #J_qwp.quarter_waveplate(azimuth=pi/4)

    Jin_c = jones_vector.create_Jones_vectors('Jin_c')
    Jin_c.circular_light(kind='l')
    #Jin_c.draw_ellipse()
    #plt.show()

    Jin_l = jones_vector.create_Jones_vectors('Jin_l')
    Jin_l.linear_light()
    #Jin_l.draw_ellipse()

    #J_ideal_out = J_qwp*J_qwp*Jin_c

    #J_ideal_out.draw_ellipse()
    #plt.show()
    #exit('hello : )')

    #Jin_c.draw_ellipse()
    #plt.show()
    #Jin.draw_ellipse()
    Jout_l = J * Jin_l
    Jout_c = J * Jin_c
    print(len(Jout_l))
    pe_export(f, Jout_l, name='x_cl4_02_15_n6')

    #plt.plot(int_x)
    #plt.plot(int_y)
    #plt.show()
    #Jout = Jhi * Jin
    #Jout[::3].draw_ellipse()
    #Jout.normalize()
    #print(Jout.parameters.delay())
    #print(Jout.parameters)
    #plt.show()
    #Jout_l.draw_ellipse()
    #plt.show()
    #Jout_c.draw_ellipse()
    #plt.show()
    #Jout.draw_ellipse()
    #plt.show()

    #print(Jout.parameters)

    #A, B = j[:, 0, 0], j[:, 0, 1]
    #res_mass = 2 * arctan(sqrt((A.imag ** 2 + B.imag ** 2) / (A.real ** 2 + B.real ** 2)))

    #Jhwpi = jones_matrix.create_Jones_matrices()
    #Jhwpi.half_waveplate(azimuth=pi/4)

    #Jlin = jones_vector.create_Jones_vectors()
    #Jlin.linear_light()

    #Jhi = jones_matrix.create_Jones_matrices()
    #Jhi.retarder_linear(R=res_mass)

    #Jo = Jhwpi*Jlin

    #plt.plot(2*pi-Jo.parameters.delay())
    #plt.plot(res_mass)
    #plt.show()

    #Jo.draw_ellipse()
    #plt.show()
    #plt.plot(res_mass)
    #plt.show()
    #plt.plot(2*(Jout.parameters.delay()-pi)/pi)
    #plt.plot(2*delt_min / pi, label='delt min')
    #plt.plot(2*delt/pi, label='wu chipman')
    #plt.plot(2*res_mass/pi, label='masson')
    #plt.legend()
    #plt.show()

