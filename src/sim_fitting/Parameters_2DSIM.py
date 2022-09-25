import numpy as np

class parameters(object):
    # set the parameters for getModamp
    def __init__(self, Ny, Nx, wave_length, excNA, setup=0):
        self.setupNUM = setup
        self.lamda = wave_length
        self.Nx = Nx
        self.Ny = Ny
        self.constbkgd = 100.0
        self.dx = 62.6e-3
        self.dy = 62.6e-3
        self.dxy = self.dx
        self.dkx = 1 / (self.Nx * self.dx)
        self.dky = 1 / (self.Ny * self.dy)
        self.dkr = np.min([self.dkx, self.dky])
        self.nphases = 3
        self.ndirs = 3
        self.scale = 1
        self.norders = int((self.nphases + 1) / 2)
        self.napodize = 10
        self.recalcarrays = 2
        self.ifshowmodamp = 0

        if self.setupNUM == 1:
            self.k0angle_c = [-1.66, -0.6128, 0.4344]
            self.k0angle_g = [3.2269, 2.1797, 1.1325]
        elif self.setupNUM == 0:
            self.k0angle_c = [1.48, 2.5272, 3.5744]
            self.k0angle_g = [0.0908, -0.9564, -2.0036]  # 1.48 - pi/2
        elif self.setupNUM == 2:
            self.k0angle_c = [1.5708, 2.618, 3.6652]
            self.k0angle_g = [0, -1.0472, -2.0944]  # 1.48 - pi/2

        self.detecNA = 1.3
        self.NA = excNA
        self.space = self.lamda / self.NA / 2
        self.k0mod = 1 / self.space

