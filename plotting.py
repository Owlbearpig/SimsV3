import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import (array, asarray, cos, exp, linspace, matrix, meshgrid,
                   ndarray, ones, outer, real, remainder, sin, size, sqrt,
                   zeros_like, arccos, arcsin, arctan, arctan2, pi, dot, argmin)
from scipy.signal import fftconvolve
from py_pol import stokes, mueller, jones_vector, jones_matrix

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


def draw_empty_sphere(ax, angle_view, axis_equal=True):
    """Fucntion that plots an empty Poincare sphere.
    """

    elev, azim = angle_view
    max_size = 1

    u = np.linspace(0, 360 * degrees, 90)
    v = np.linspace(0, 180 * degrees, 90)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x,
                    y,
                    z,
                    rstride=4,
                    cstride=4,
                    color='b',
                    edgecolor="k",
                    linewidth=.0,
                    alpha=0.1)
    ax.plot(np.sin(u),
            np.cos(u),
            0,
            color='k',
            linestyle='dashed',
            linewidth=0.5)
    ax.plot(np.sin(u),
            np.zeros_like(u),
            np.cos(u),
            color='k',
            linestyle='dashed',
            linewidth=0.5)
    ax.plot(np.zeros_like(u),
            np.sin(u),
            np.cos(u),
            color='k',
            linestyle='dashed',
            linewidth=0.5)

    ax.plot([-1, 1], [0, 0], [0, 0], 'k-.', lw=1, alpha=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], 'k-.', lw=1, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-1, 1], 'k-.', lw=1, alpha=0.5)

    ax.plot(xs=(1, ),
            ys=(0, ),
            zs=(0, ),
            color='black',
            marker='o',
            markersize=4,
            alpha=0.5)

    ax.plot(xs=(-1, ),
            ys=(0, ),
            zs=(0, ),
            color='black',
            marker='o',
            markersize=4,
            alpha=0.5)
    ax.plot(xs=(0, ),
            ys=(1, ),
            zs=(0, ),
            color='black',
            marker='o',
            markersize=4,
            alpha=0.5)
    ax.plot(xs=(0, ),
            ys=(-1, ),
            zs=(0, ),
            color='black',
            marker='o',
            markersize=4,
            alpha=0.5)
    ax.plot(xs=(0, ),
            ys=(0, ),
            zs=(1, ),
            color='black',
            marker='o',
            markersize=4,
            alpha=0.5)
    ax.plot(xs=(0, ),
            ys=(0, ),
            zs=(-1, ),
            color='black',
            marker='o',
            markersize=4,
            alpha=0.5)
    distance = 1.15
    ax.text(distance, 0, 0, 'Q', fontsize=14)
    ax.text(0, distance, 0, 'U', fontsize=14)
    ax.text(0, 0, distance, 'V', fontsize=14)

    ax.view_init(elev=elev / degrees, azim=azim / degrees)

    ax.set_xlabel('$S_1$', fontsize=14, labelpad=-10)
    ax.set_ylabel('$S_2$', fontsize=14, labelpad=-10)
    ax.set_zlabel('$S_3$', fontsize=14, labelpad=-10)
    ax.grid(False)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xlim(-max_size, max_size)
    ax.set_ylim(-max_size, max_size)
    ax.set_zlim(-max_size, max_size)

    plt.tight_layout()
    # set_aspect_equal_3d(ax)
    if axis_equal:
        try:
            ax.set_aspect('equal')
        except:
            print(
                'Axis equal not supported by your current version of Matplotlib'
            )

def draw_stokes_points(ax,
                       S,
                       I_min=None,
                       I_max=None,
                       D_min=None,
                       D_max=None,
                       normalize=True,
                       remove_depol=False,
                       kind='scatter',
                       color_scatter='Intensity',
                       color_line='r',
                       label=None):
    """Function to draw Stokes vectors on the poincare sphere.

    Parameters:
        S (Stokes): Stokes object.
        normalize (bool): If True, normalize the Stokes vectors to have intensity 1. Default: True.
        I_min (float): Minimum intensity among all subplots.
        I_max (float): Amximum intensity among all subplots.
        D_min (float): Minimum polarization degree among all subplots.
        D_max (float): Amximum polarization degree among all subplots.
        remove_depol (bool): If True, plot the polarized part of the Stokes vector. Default: False.
        kind (str): Plot type: LINE, SCATTER or BOTH. Default: SCATTER.
        color_scatter (str): There are three options. INTENSITY sets the color of the points intensity dependent. DEGREE sets the color to match the polarization degree. Another posibility is to use valid color strings such as 'r'. Default: INTENSITY.
        color_line (str): Only valid color strings such as 'r'. Default: 'r'.

    Returns:

    """
    # Avoid empty objects
    if S is None or S.size == 0:
        return None

    # if there is just one item in the Stokes object, change to scatter
    if S.size == 1:
        kind = 'scatter'

    # Remove deopolarization if required
    if remove_depol:
        Sp, _ = S.parameters.polarized_unpolarized()
    else:
        Sp = S

    # Normalize to intensity=1 if required
    I = Sp.parameters.intensity(shape=False, out_number=False)
    if normalize:
        Sn = Sp.normalize(keep=True)
    else:
        Sn = Sp

    # Calculate all the required parameters
    _, S1, S2, S3 = Sn.parameters.components(shape=False, out_number=False)
    im_scatter = None
    if color_scatter.upper() == 'DEGREE':
        depol = Sn.parameters.degree_depolarization(shape=False,
                                                    out_number=False)

    # Make the plots
    if kind.upper() in ('SCATTER', 'BOTH'):
        if color_scatter.upper() == 'INTENSITY':
            im_scatter = ax.scatter(S1,
                                    S2,
                                    S3,
                                    c=I,
                                    s=60,
                                    vmin=I_min,
                                    vmax=I_max)
        elif color_scatter.upper() == 'DEGREE':
            im_scatter = ax.scatter(S1,
                                    S2,
                                    S3,
                                    c=depol,
                                    s=60,
                                    vmin=D_min,
                                    vmax=D_max)
        else:
            ax.scatter(S1, S2, S3, c=color_scatter, s=60, label=label)

    if kind.upper() in ('LINE', 'BOTH') and Sn.size > 1:
        ax.plot(S1, S2, S3, c=color_line, lw=2, label=label)

    return im_scatter

def draw_poincare(S,
                  normalize=True,
                  remove_depol=False,
                  kind='scatter',
                  color_scatter='r',
                  color_line='r',
                  angle_view=[0.5, -1],
                  figsize=(6, 6),
                  filename='',
                  subplots=None,
                  axis_equal=True):
    """Function to draw the poincare sphere.
    It admits Stokes or a list with Stokes, or None

    Parameters:
        stokes_points (Stokes, list, None): list of Stokes points.
        angle_view (float, float): Elevation and azimuth for viewing.
        is_normalized (bool): normalize intensity to 1.
        kind (str): 'line' or 'scatter'.
        color (str): color of line or scatter.
        label (str): labels for drawing.
        filename (str): name of filename to save the figure.
        axis_equal (bool): If True, the axis_equal method is used. Default: True.
    """
    # Calculate the number of subplots
    if subplots is None:
        Nx, Ny, Nsubplots, Ncurves = (1, 1, 1, S.size)
    elif type(subplots) is tuple:
        Nsubplots = np.prod(np.array(subplots[0:2]))
        if S.size % Nsubplots != 0:
            raise ValueError(
                'Shape {} is not valid for the object {} of {} elements'.
                format(subplots, S.name, S.size))
        Nx, Ny = subplots[0:2]
        Ncurves = int(S.size / Nsubplots)
    elif subplots.upper() == 'INDIVIDUAL':
        Ny = int(np.floor(np.sqrt(S.size)))
        Nx = int(np.ceil(S.size / Ny))
        Nsubplots = S.size
        Ncurves = 1
    elif subplots.upper() == 'AS_SHAPE':
        if S.ndim < 2:
            Nx, Ny = (1, S.size)
            Nsubplots, Ncurves = (S.size, 1)
        else:
            Nx, Ny = S.shape[0:2]
            Nsubplots = Nx * Ny
        Ncurves = int(S.size / Nsubplots)
    else:
        raise ValueError('{} is not a valid subplots option.')

    # Flatten the object
    Sf = S.copy()
    Sf.shape = np.array([Sf.size])

    # Calculate color min and max values
    if color_line.upper() == 'INTENSITY' or color_scatter.upper(
    ) in 'INTENSITY':
        I = Sf.parameters.intensity(out_number=False)
        I_min, I_max = (I.min(), I.max())
    else:
        I_min, I_max = (None, None)
    if color_line.upper() == 'DEGREE' or color_scatter.upper() in 'DEGREE':
        pol = Sf.parameters.degree_polarization(out_number=False)
        D_min, D_max = (pol.min(), pol.max())
    else:
        D_min, D_max = (None, None)

    # Create the figure
    fig = plt.figure(figsize=figsize)
    ax = []

    # Loop to generate the poincare spheres
    for indS in range(Nsubplots):
        # Divide in subplots
        axis = fig.add_subplot(Nx,
                               Ny,
                               indS + 1,
                               projection='3d',
                               adjustable='box')
        ax += [axis]
        # Create the Poincare sphere
        draw_empty_sphere(axis, angle_view=angle_view, axis_equal=axis_equal)

        # Add points from Stokes
        im_scatter = draw_stokes_points(axis,
                                        Sf[indS * Ncurves:(indS + 1) *
                                           Ncurves],
                                        I_min=I_min,
                                        I_max=I_max,
                                        D_min=D_min,
                                        D_max=D_max,
                                        normalize=normalize,
                                        remove_depol=remove_depol,
                                        kind=kind,
                                        color_scatter=color_scatter,
                                        color_line=color_line)
        # Add titles
        if Nsubplots > 1:
            if subplots in ('individual', 'Individual', 'INDIVIDUAL'):
                string = str(indS)
            else:
                string = str(list(np.unravel_index(indS, (Nx, Ny))))
            plt.title(string, fontsize=18)
        else:
            plt.title(S.name, fontsize=26)

    # Add supertitle if required
    if Nsubplots > 1:
        fig.suptitle(S.name, fontsize=26)

    # Add colormap if required
    if im_scatter is not None:
        cbar = fig.colorbar(im_scatter, ax=ax)
        if color_scatter.upper() == 'INTENSITY':
            cbar.set_label(label='Intensity', fontsize=14)
        else:
            cbar.set_label(label='Pol. degree', fontsize=14)

    return ax, fig

def stokes_path(s1, s2):
    # calc. great arc path in cartesian coords. between stokes params s1, s2
    # s1,2 : stokes params, (1,x,y,z) on poincare
    # Longitude = azimuth = phi (I think)
    # https://math.stackexchange.com/questions/383711/parametric-equation-for-great-circle
    t = np.linspace(0, 1, 25)
    s1_array, s2_array = s1.M.flatten(), s2.M.flatten()

    x1, y1, z1 = s1_array[1:]
    x2, y2, z2 = s2_array[1:]

    lon1, lon2 = arctan2(y1, x1), arctan2(y2, x2)
    lat1, lat2 = arctan2(z1, sqrt(x1**2+y1**2)), arctan2(z2, sqrt(x2**2+y2**2))

    d = arccos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))
    A = sin((1 - t) * d) / sin(d)
    B = sin(t * d) / sin(d)

    x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon2)
    y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon2)
    z = A * sin(lat1) + B * sin(lat2)

    S = stokes.create_Stokes()
    stokes_vectors = (np.ones_like(t), x, y, z)
    S.from_components(stokes_vectors)

    return S

def poincare_path(J0, T_matrix_list):

    fig = plt.figure(figsize=(6, 6))

    axis = fig.add_subplot(1, 1, 1, projection='3d', adjustable='box')
    draw_empty_sphere(axis, angle_view=[0.5, -1], axis_equal=True)

    prev_J = J0
    for i, T in enumerate(T_matrix_list):
        current_T = jones_matrix.create_Jones_matrices()
        current_T.from_matrix(T)

        current_J = current_T*prev_J

        S_current = stokes.create_Stokes()
        S_current.from_Jones(current_J)

        S_prev = stokes.create_Stokes()
        S_prev.from_Jones(prev_J)

        if i == 0:
            print(S_prev)
            draw_stokes_points(axis, S_prev, I_min=1, I_max=1, D_min=1, D_max=1, normalize=True,
                               remove_depol=False,
                               kind='scatter',
                               color_scatter='black',
                               color_line='black',
                               label='Input state')


        path = stokes_path(S_prev, S_current)
        draw_stokes_points(axis, path, I_min=1, I_max=1, D_min=1, D_max=1, normalize=True,
                           remove_depol=False,
                           kind='line',
                           color_scatter='r',
                           color_line=name_colors[i],
                           label=f'Waveplate {i+1}')

        if i != 0:
            draw_stokes_points(axis, S_prev, I_min=1, I_max=1, D_min=1, D_max=1, normalize=True,
                               remove_depol=False,
                               kind='scatter',
                               color_scatter=name_colors[i],
                               color_line='r')

        prev_J = current_J

    print(S_current)

    draw_stokes_points(axis, S_current, I_min=1, I_max=1, D_min=1, D_max=1, normalize=True,
                       remove_depol=False,
                       kind='scatter',
                       color_scatter=name_colors[-1],
                       color_line='r', label='Output state')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    from results import *
    from py_pol import jones_vector
    from functions import setup

    Jin_l = jones_vector.create_Jones_vectors('Jin_l')
    Jin_l.linear_light()

    res = result_GHz

    j_individual, freqs, wls = setup(res, return_vals=True, return_individual=True)

    freq_idx = int(len(freqs) // 1.0001)

    print(freqs[freq_idx]*10**-9)
    j_individual = j_individual[freq_idx]

    j_individual = flip(j_individual)

    poincare_path(Jin_l, j_individual)
