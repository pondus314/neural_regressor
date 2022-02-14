from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import torch


class Visualiser2D:

	def __init__(self):
		self.fig = plt.figure()
		self.ax = self.fig.gca(projection='3d')
		plt.ion()
		plt.show()


	def clear(self):
		plt.cla()


	def update(self, func, lim=[-1,1,-1,1], step=0.1, transparent=False, dim=2, other_dim_val=0.):

		x = np.arange(lim[0],lim[1],step)
		y = np.arange(lim[2],lim[3],step)
		x, y = np.meshgrid(x, y)
		x = torch.tensor(x)
		y = torch.tensor(y)
		v = torch.stack([x,y], dim=2).view(x.shape[0]*x.shape[1], 2).float()
		v = torch.cat((torch.ones(v.shape[0], dim-v.shape[1]).float() * other_dim_val, v), dim=1)
		z = func(v)
		z = z.view(x.shape).detach().numpy()

		if transparent:
			self.ax.plot_wireframe(x, y, z)
		else:
			self.ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=0)

		plt.draw()
		plt.pause(0.01)

