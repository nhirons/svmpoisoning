import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import AdversarialAgent

class Plotter(AdversarialAgent.AdversarialAgent):

    def __init__(self):
        self.random_state = np.random.RandomState(19680801)

    def plot_3d_loss_surface(self):
        npts = 401
        x1 = random_state.uniform(-200, 200, npts)
        x2 = random_state.uniform(-200, 200, npts)
        xc_ = np.array(list(zip(x1, x2)))
        z_ = np.array([loss_function_SVM(X, y, \
            xc_[i,:][np.newaxis, :],1, X2,y2) for i in range(npts)])
        # define grid.
        x1i = np.linspace(-150, 150, 100)
        x2i = np.linspace(-150, 150, 100)
        XX, YY = np.meshgrid(x1i, x2i)
        # grid the data.
        zi = griddata(x1, x2, z_, x1i, x2i, interp='linear')

        fig = plt.figure(figsize = (18,18))

        for i in range(1,10):
            ax = fig.add_subplot(3, 3, i, projection='3d')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('Hinge loss')
            ax.plot_surface(XX, YY, zi, cmap = 'viridis')
            ax.view_init(30, (i-1)*40)

        plt.show()

    def plot_2d_loss_surface(self):
        npts = 1000
        x1 = random_state.uniform(-60, 60, npts)
        x2 = random_state.uniform(-60, 100, npts)
        xc_ = np.array(list(zip(x1, x2)))
        z_ = np.array([loss_function_SVM(X, y, \
            xc_[i,:][np.newaxis, :],1, X2,y2) for i in range(npts)])
        # define grid.
        x1i = np.linspace(-50, 50, 100)
        x2i = np.linspace(-50, 100, 200)
        # grid the data.
        zi_ = griddata(x1, x2, z_, x1i, x2i, interp='linear')
        # contour the gridded data
        CS = plt.contour(x1i, x2i, zi_, 15)
        CS = plt.contourf(x1i, x2i, zi_, 15,
                          vmax=zi_.max(), vmin=zi_.min())
        plt.colorbar()  # draw colorbar
        # plot data points.
        plt.plot(self.Xp[:,0],self.Xp[:,1],'bo')
        plt.plot(self.Xn[:,0],self.Xn[:,1],'ro')
        #plt.scatter(X[:,0], X[:,1],marker='o', s=5, zorder=10)
        plt.xlim(-50, 50)
        plt.ylim(-50, 100)
        plt.title('Hinge loss surface')
        plt.show()

