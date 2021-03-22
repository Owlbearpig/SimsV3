
""" 8 (C)
d = np.array([570.1, 379.8, 379.8, 379.8, 379.8, 379.8, 379.8, 570.1])
angles_C = np.array([303.9, 263.4, 330.2, 344.9, 29.8, 15.4, 308.3, 347.7])

x0 = np.concatenate((d, angles_C))

bounds = [
    (570,571),(379,380),(379,380),(379,380),(379,380),(379,380),(379,380),(570,571),
    (303, 304),(263, 264),(330, 331),(344, 345),(29, 30), (15, 16), (308, 309), (347, 348)]
"""
# 6 A
bounds = [
    #(565.95,566.05),(377.35,377.45),(377.35,377.45),(377.35,377.45),(377.35,377.45),(377.35,377.45),
    #(12.25, 12.35),(56.65, 56.75),(28.05, 28.15),(356.45, 356.55),(301.95, 302.05), (349.15, 349.25)]
    (12.2, 12.4),(56.6, 56.8),(28.0, 28.2),(356.4, 356.6),(301.9, 302.1), (349.1, 349.3)]

d = np.array([566.0, 377.4, 377.4, 377.4, 377.4, 377.4])
angles_A = np.array([12.3, 56.7, 28.1, 356.5, 302.0, 349.2])

#x0 = np.concatenate((d, angles_A))
x0 = angles_A
print(erf(x0))

#local_min_kwargs = {'ftol': 10**-6, 'gtol': 10**-6}
#ret = minimize(erf, x0, method='L-BFGS-B', bounds=bounds, options=local_min_kwargs)

minimizer_kwargs = {"method":"L-BFGS-B", "bounds": bounds}
xmin = [t[0] for t in bounds]
xmax = [t[1] for t in bounds]


class MyBounds(object):
    def __init__(self):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


class MyTakeStep(object):
   def __init__(self, stepsize=0.5):
       self.stepsize = stepsize
   def __call__(self, x):
       s = self.stepsize
       x[0] += np.random.uniform(-2.*s, 2.*s)
       x[1:] += np.random.uniform(-s, s, x[1:].shape)
       return x

mybounds = MyBounds()
mytakestep = MyTakeStep()
#ret = basinhopping(erf, x0, minimizer_kwargs=minimizer_kwargs, niter=1000, accept_test=mybounds, callback=print_fun)
ret = basinhopping(erf, x0, niter=1000, accept_test=mybounds, callback=print_fun, take_step=mytakestep)

print(ret)
