import skrf as rf
import matplotlib.pyplot as plt
import numpy as np
from generate_plotdata import export_csv
import os
from scipy.constants import c as c0

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

path = r'E:\CURPROJECT\SimsV3\GHz\CST_SWP_HIPS_dSweep'

touchstoneFiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(path)
               for name in files
               if name.endswith('.s4p')]

bf = np.array([])
freqs = np.array([])
res_tuples = []
for file in touchstoneFiles:
    hips_waveguide = rf.Network(file)

    f = hips_waveguide.f
    ampl = 20*np.log10(np.abs(hips_waveguide.s[:,2,0]))
    headerpart1 = hips_waveguide.comments.split('d=')[1]
    d = float(headerpart1.split(';')[0])


    res_tuples.append((d, ampl))

    freq = f[np.argmin(ampl)]
    freqs = np.append(freqs, freq)

    d = d*10**-3

    bf = np.append(bf, c0/(2*freq*d))

datapoints = zip(freqs, bf)
datapoints_filtered = []
for datapoint in datapoints:
    if datapoint[0]/10**9 < 80 and datapoint[1] < 0.08175:
        continue
    if datapoint[0]/10**9 > 119 and datapoint[1] > 0.0706:
        continue
    datapoints_filtered.append(datapoint)

datapoints_filtered.sort(key=lambda x: x[0])

datapoints_filtered = np.array(datapoints_filtered)

#plt.plot(datapoints_filtered[:,0]/10**9, datapoints_filtered[:,1], label='birefringence CST')
#plt.scatter(freqs/10**9, bf, label='birefringence CST')
#export_csv({'freqs': datapoints_filtered[:,0]/10**9, 'bf': datapoints_filtered[:,1]}, 'birefringence_CST.csv')


res_tuples.sort(key=lambda x: x[0])
res_tuples = np.array(res_tuples)
indices = [0, 17, 33, 46, 59, 75, 93, 107, 124, 141, 158]
data_export = {}
for i, res in enumerate(res_tuples):
    if i in indices:
        if int(round(res[0], 0)) > 24:
            continue
        if int(round(res[0], 0)) < 18:
            continue
        label = str(int(round(res[0], 0)))
        plt.plot(f, res[1], label=label)
        data_export[label] = res[1]

data_export['freqs'] = f

export_csv(data_export, 'cst_bf_plot_data.csv')

plt.legend()
plt.show()
y = datapoints_filtered[:,1]
#yhat = savitzky_golay(y, 71, 3) # window size 51, polynomial order 3

plt.plot(datapoints_filtered[:,0]/10**9, y, label='y')
#plt.plot(datapoints_filtered[:,0]/10**9, yhat, color='red', label='yhat')

plt.legend()
plt.show()
