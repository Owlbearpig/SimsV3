import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import (array, asarray, cos, exp, linspace, matrix, meshgrid,
                   ndarray, ones, outer, real, remainder, sin, size, sqrt,
                   zeros_like, arccos, arcsin, arctan)
from scipy.signal import fftconvolve
from py_pol.stokes import draw_poincare
from py_pol import stokes

degrees = np.pi / 180

Axes3D = Axes3D  # pycharm auto import
colors = matplotlib.colors.TABLEAU_COLORS
name_colors = list(colors)
linestyles = [('dashdot', 'dashdot'),
              ('loosely dashdotted', (0, (3, 10, 1, 10))),
              ('dashdotted', (0, (3, 5, 1, 5))),
              ('densely dashdotted', (0, (3, 1, 1, 1))),
              ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
              ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
              ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


def draw_ellipse(E,
                 N_angles=91,
                 filename='',
                 figsize=(6, 6),
                 limit='',
                 draw_arrow=True,
                 depol_central=False,
                 depol_contour=False,
                 depol_prob=False,
                 subplots=None,
                 N_prob=256,
                 contour_levels=(0.9, ),
                 cmap='hot',
                 return_values=False):
    """Draws polarization ellipse of Jones vector.

    Parameters:
        E (Jones_vector or Stokes): Light object.
        N_angles (int): Number of angles to plot the ellipses. Default: 91.
        limit (float): limit for drawing. If empty, it is obtained from ampltiudes.
        filename (str): name of filename to save the figure.
        figsize (tuple): A tuple of length 2 containing the figure size. Default: (8,8).
        draw_arrow (bool): If True, draws an arrow containing the turning sense of the polarization. Does not work with linear polarization vectors. Default: True.
        depol_central (bool): If True, draws a central circle containing the unpolarized field amplitude. Default: False.
        depol_contour (bool): If True, draws a line enveloping the polarization ellipse in ordeer to plot the depolarization. Default: False.
        depol_dist (bool): If True, plots the probability distribution of the electric field. Default: False.
        subplots (string, tuple or None): If AS_SHAPE, divides the figure in several subplots as the shape of the py_pol object. If INDIVIDUAL, each vector is represented in its own subaxis, trying to use a square grid. If tuple, divides the figure in that same number of subplots. If None, all ellipses are plot in the same axes. Default: None.
        N_prob (int): Number of points in each dimension for probability distributions. Default: 256.
        prob (flota, np.ndarray, tuple or list): Contains the contour levels (normalized to 1). Default: 0.9.
        cmap (str or color object): Default colormap for probability distributions. Default: hot.

    Returns:
        fig (handle): handle to figure.
        ax (list of handles): handles to axes.
    """
    # Calculate the electric field amplitudes and the delays
    if E._type == 'Jones_vector':
        E0x, E0y = E.parameters.amplitudes(shape=False)
        E0u = np.zeros(1)
    else:
        E0x, E0y, E0u = E.parameters.amplitudes(shape=False, give_unpol=True)
    delay = E.parameters.delay(shape=False)
    phase = E.parameters.global_phase(shape=False)
    if phase is None:
        phase = np.zeros_like(E0x)
    if np.isnan(phase).any():
        phase[np.isnan(phase)] = 0
    # Create the angle variables
    angles = linspace(0, 360 * degrees, N_angles)
    Angles, E0X = np.meshgrid(angles, E0x)
    _, E0Y = np.meshgrid(angles, E0y)
    _, Delay = np.meshgrid(angles, delay)
    _, Phase = np.meshgrid(angles, phase)
    if E._type == 'Jones_vector':
        is_linear = E.checks.is_linear(shape=False, out_number=False)
    else:
        is_linear = E.checks.is_linear(shape=False,
                                       out_number=False,
                                       use_nan=False)
    # Create the electric field distributions
    Ex = E0X * np.cos(Angles + Phase)
    Ey = E0Y * np.cos(Angles + Phase + Delay)
    # Calculate the depolarization central distribution
    if E._type == 'Stokes' and depol_central:
        _, E0U = np.meshgrid(angles, E0u)
        Exu = E0U * np.cos(Angles)
        Eyu = E0U * np.sin(Angles)
    # Safety arrays
    if E._type == 'Stokes':
        is_pol = E.checks.is_polarized(shape=False, out_number=False)
        is_depol = E.checks.is_depolarized(shape=False, out_number=False)
    else:
        if E.size < 2:
            is_pol = np.array([True])
        else:
            is_pol = np.ones_like(E0x).flatten()
    # Set automatic limits
    if limit in [0, '', [], None]:
        if depol_contour or depol_prob:
            limit = np.array([E0x.max() + E0u.max(),
                              E0y.max() + E0u.max()]).max() * 1.2
        else:
            limit = np.array([E0x.max(), E0y.max(), E0u.max()]).max() * 1.2

    # Prepare the figure and the subplots
    if not return_values:
        fig = plt.figure(figsize=figsize)
    if depol_prob:
        if type(subplots) is tuple and E.size == np.prod(np.array(subplots)):
            pass  # Only case subplots is not overwritten
        else:
            subplots = 'individual'
    if subplots is None:
        # Just one subplot
        Nx, Ny, Nsubplots, Ncurves = (1, 1, 1, E.size)
    elif type(subplots) is tuple:
        # Set number of subplots
        Nsubplots = np.prod(np.array(subplots[0:2]))
        if E.size % Nsubplots != 0:
            raise ValueError(
                'Shape {} is not valid for the object {} of {} elements'.
                format(subplots, E.name, E.size))
        Ncurves = E.size / Nsubplots
        Nx, Ny = subplots[0:2]
        indS, indE = (0, 0)
    elif subplots in ('AS_SHAPE', 'as_shape', 'As_shape'):
        # Subplots given by phase
        if E.ndim < 2:
            Nx, Ny = (1, E.size)
            Nsubplots, Ncurves = (E.size, 1)
        else:
            Nx, Ny = E.shape[0:2]
            Nsubplots, Ncurves = (Nx * Ny, E.size / (Nx * Ny))
        indS, indE = (0, 0)
    elif subplots in ('individual', 'Individual', 'INDIVIDUAL'):
        Ny = int(np.floor(np.sqrt(E.size)))
        Nx = int(np.ceil(E.size / Ny))
        Nsubplots, Ncurves = (E.size, 1)
    else:
        raise ValueError('{} is not a valid subplots option.')
    # If contour lines or probability must be plotted, calculate the probability distributions and linestyles
    if depol_contour or depol_prob:
        # Create the basic probability distribution
        x = np.linspace(-limit, limit, N_prob)
        X, E0U, Y = np.meshgrid(x, E0u, x)
        prob = np.exp(-(X**2 + Y**2) / (E0U**2))
        # Create the ellipse distribution
        indX = np.abs(np.subtract.outer(x, Ex)).argmin(0).flatten()
        indY = np.abs(np.subtract.outer(x, Ey)).argmin(0).flatten()
        indE = np.repeat(np.arange(E.size), N_angles)
        # indE = np.flip(indE)
        ellipse_3D = zeros_like(X, dtype=float)
        ellipse_3D[indE, indY, indX] = 1
        # Convolute them adn normalize to 1
        prob = fftconvolve(ellipse_3D, prob, mode='same', axes=(1, 2))
        _, MAX, _ = meshgrid(x, prob.max(axis=(1, 2)), x)
        prob = prob / MAX
        # Remove info for totally polarized vectors
        prob[~is_depol, :, :] = 0
        # Linestyles
        if len(contour_levels) <= len(linestyles):
            line_styles = linestyles[:len(contour_levels)]
        else:
            line_styles = linestyles[0]

    # Main loop
    ax = []
    for ind in range(E.size):  # Loop in curves
        if not return_values:
            # Initial considerations for the subplot
            indS = int(np.floor(ind / Ncurves))
            indC = int(ind % Ncurves)
            if indC == 0:
                axis = fig.add_subplot(Nx, Ny, indS + 1)
                ax.append(axis)
                if Nsubplots > 1:
                    if subplots in ('individual', 'Individual', 'INDIVIDUAL'):
                        string = str(indS)
                    else:
                        string = str(list(np.unravel_index(indS, (Nx, Ny))))
                    plt.title(string, fontsize=18)
                else:
                    plt.title(E.name, fontsize=26)
            # Other considerations
            if depol_prob:
                color = 'w'
            else:
                color = colors[name_colors[ind % 10]]
            if subplots in ('AS_SHAPE', 'as_shape',
                            'As_shape') and Nx * Ny > 1 and Ncurves > 1:
                string = str(list(np.unravel_index(ind, E.shape)[2:]))
            else:
                if Ncurves == 1:
                    string = 'Polarized'
                else:
                    string = str(list(np.unravel_index(ind, E.shape)))
            # Plot the probability distribution
            if depol_prob and is_depol[ind]:
                IDimage = axis.imshow(prob[ind, :, :],
                                      interpolation='bilinear',
                                      aspect='equal',
                                      origin='lower',
                                      extent=[-limit, limit, -limit, limit])
                # axis = axis[0]
        # Plot the curve
        if return_values:
            return Ex, Ey
        if is_pol[ind]:
            axis.plot(Ex[ind, :], Ey[ind, :], lw=2, label=string, color=color)
            if draw_arrow and ~is_linear[ind]:
                axis.arrow(Ex[ind, 0],
                           Ey[ind, 0],
                           Ex[ind, 4] - Ex[ind, 0],
                           Ey[ind, 4] - Ey[ind, 0],
                           width=0,
                           head_width=0.075 * limit,
                           linewidth=0,
                           color=color,
                           length_includes_head=True)
        elif E._type == 'Stokes' and depol_central and ~is_depol[ind]:
            axis.plot(np.zeros(2),
                      np.zeros(2),
                      lw=2,
                      label=string,
                      color=color)
        elif depol_central or depol_prob:
            print('Field {} is empty.'.format(string))
        else:
            print('Field {} is empty or totally depolarized.'.format(string))
        # Add the depolarization for Stokes vectors
        if E._type == 'Stokes' and depol_central and is_depol[ind]:
            axis.plot(Exu[ind, :], Eyu[ind, :], lw=1.5, color=color, ls='--')
        if E._type == 'Stokes' and depol_contour and is_pol[ind] and is_depol[
                ind]:
            CS = axis.contour(
                x,
                x,
                prob[ind, :, :],
                contour_levels,
                colors=(color),
                linewidths=1.5,
                # linestyles=line_styles)
                linestyles=('dashdot'))
        # Additions to figure
        if indC == Ncurves - 1:
            plt.axis('equal')
            plt.axis('square')
            plt.grid(True)
            axis.set_xlim(-limit, limit)
            axis.set_ylim(-limit, limit)
            axis.set_xlabel('$E_x (V/m)$', fontsize=14)
            axis.set_ylabel('$E_y (V/m)$', fontsize=14)
            plt.tight_layout()
            if Ncurves > 1:
                plt.legend()
            elif depol_contour and indC == Ncurves - 1:
                for ind, elem in enumerate(contour_levels):
                    CS.collections[ind].set_label('P = {}'.format(elem))
                plt.legend()
            if depol_prob and is_depol[indS]:
                divider = make_axes_locatable(axis)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(IDimage, cax=cax)
                IDimage.set_cmap(cmap)
    if Nsubplots > 1:
        fig.suptitle(E.name, fontsize=26)
    # Save the image if required
    if filename not in (None, [], ''):
        plt.savefig(filename)
        print('Image {} saved succesfully!'.format(filename))
    return fig, ax

def path(s1, s2):
    # calc. great arc path in cartesian coords. between stokes params s1, s2
    # s1,2 : stokes params, (1,x,y,z) on poincare
    # Longitude = azimuth = phi (I think)
    # https://math.stackexchange.com/questions/383711/parametric-equation-for-great-circle
    t = np.linspace(0, 1, 100)

    x1, y1, z1 = s1[1:]
    x2, y2, z2 = s2[1:]

    lat1, lat2 = arccos(z1), arccos(z2)
    lon1, lon2 = arctan(y1/x1), arctan(y2/x2)

    d = arccos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))
    A = sin((1 - t) * d) / sin(d)
    B = sin(t * d) / sin(d)

    x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon2)
    y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon2)
    z = A * sin(lat1) + B * sin(lat2)

    stokes_vectors = []
    for i in range(len(t)):
        S = stokes.create_Stokes()
        stokes_tuple = (1, x[i], y[i], z[i])
        stokes_vectors.append(S.from_components(stokes_tuple))

    return stokes_vectors

S_top = stokes.create_Stokes()
S_top.circular_light()
S_top.draw_poincare()
plt.show()
S_top = array(S_top)
print(S_top)
S_middle = stokes.create_Stokes()
S_middle.linear_light()
S_middle.draw_poincare()
plt.show()


