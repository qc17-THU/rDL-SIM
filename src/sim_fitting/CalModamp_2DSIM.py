import numpy as np
import numpy.fft as F
import glob
import scipy
import tensorflow as tf
import math
from .Parameters_2DSIM import parameters
from scipy.interpolate import interp1d
# from ..utils.read_mrc import read_mrc


def make_matrix(nphases, norders):
    sepMatrix = np.zeros((2*norders-1, nphases))
    phi = 2 * np.pi / nphases

    j = np.arange(0, nphases, 1)
    order = np.arange(1, norders, 1)

    sepMatrix[0, :] = 1.0
    sepMatrix[1, :] = np.cos(order * j * phi)
    sepMatrix[2, :] = np.sin(order * j * phi)

    return sepMatrix


def apodize(napodize, Nx, Ny, Nz, image):
    apoimage = image
    imageUp = image[0:napodize, :]
    imageDown = image[Ny-napodize:Ny, :]

    diff = (imageDown[::-1] - imageUp) / 2
    l = np.arange(0, napodize, 1)
    fact = 1 - np.sin((l + 0.5) / napodize * np.pi / 2)
    fact = np.tile(fact, (Nx, 1)).transpose()
    factor = diff * fact
    apoimage[0:napodize, :] = imageUp + factor
    apoimage[Ny-napodize:Ny, :] = imageDown - factor[::-1]

    imageLeft = apoimage[:, 0:napodize]
    imageRight = apoimage[:, Nx-napodize:Nx]
    diff = (imageRight[:, ::-1] - imageLeft) / 2
    l = np.arange(0, napodize, 1)
    fact = 1 - np.sin((l + 0.5) / napodize * np.pi / 2)
    fact = np.tile(fact, (Ny, 1))
    factor = diff * fact
    apoimage[:, 0:napodize] = imageLeft + factor
    apoimage[:, Nx-napodize:Nx] = imageRight - factor[:, ::-1]
    return apoimage


def makeoverlaps(bands, Nx, Ny, Nz, order1, order2, k0x, k0y, dxy, dz, OTF, lamda):
    otfcutoff = 0.005
    kx = k0x * (order2 - order1)
    ky = k0y * (order2 - order1)

    dkx = 1 / (Nx * dxy)
    dky = 1 / (Ny * dxy)
    dkr = np.min((dkx, dky))
    rdistcutoff = 1.35 * 2 / lamda

    if rdistcutoff > Nx / 2 * dkx:
        rdistcutoff = Nx / 2 * dkx
    if rdistcutoff > Ny / 2 * dky:
        rdistcutoff = Ny / 2 * dky

    x1 = np.arange(-Nx / 2, Nx / 2, 1) * dkx
    y1 = np.arange(-Ny / 2, Ny / 2, 1) * dky

    [X1, Y1] = np.meshgrid(x1, y1)
    rdist1 = np.sqrt(np.square(X1) + np.square(Y1))

    x12 = x1 - kx
    y12 = y1 - ky
    [X12, Y12] = np.meshgrid(x12, y12)
    rdist12 = np.sqrt(np.square(X12) + np.square(Y12))

    x21 = x1 + kx
    y21 = y1 + ky
    [X21, Y21] = np.meshgrid(x21, y21)
    rdist21 = np.sqrt(np.square(X21) + np.square(Y21))

    mask1 = (rdist1 <= rdistcutoff) * (rdist12 <= rdistcutoff)
    mask2 = (rdist1 <= rdistcutoff) * (rdist21 <= rdistcutoff)
    otflen = len(OTF)

    if order1 == 0:
        band1re = np.squeeze(bands[0, :, :])
    else:
        band1re = np.squeeze(bands[order1 * 2 - 1, :, :])
        band1im = np.squeeze(bands[order1 * 2, :, :])

    band2re = np.squeeze(bands[order2 * 2 - 1, :, :])
    band2im = np.squeeze(bands[order2 * 2, :, :])

    x = np.arange(0, otflen * dkr, dkr)
    interp = interp1d(x, OTF, kind='slinear')
    OTF1 = interp(rdist1)
    OTF2 = interp(rdist1)
    OTF1_sk0 = interp(rdist21)
    OTF2_sk0 = interp(rdist12)
    OTF1_mag = np.abs(OTF1)
    OTF2_mag = np.abs(OTF2)
    OTF1_sk0_mag = np.abs(OTF1_sk0)
    OTF2_sk0_mag = np.abs(OTF2_sk0)

    mask1 = mask1 * (OTF1_mag > otfcutoff) * (OTF2_sk0_mag > otfcutoff)
    ind_mask1 = (mask1 == 1)
    root = np.sqrt(np.square(OTF1_mag) + np.square(OTF2_sk0_mag))
    fact1 = np.zeros((Ny, Nx))
    fact1[ind_mask1] = OTF2_sk0[ind_mask1] / root[ind_mask1]
    val1re = band1re * fact1
    if order1 > 0:
        val1im = band1im * fact1
        overlap0 = val1re + 1j * val1im
    else:
        overlap0 = val1re

    mask2 = mask2 * (OTF1_sk0_mag > otfcutoff) * (OTF2_mag > otfcutoff)
    ind_mask2 = (mask2 == 1)
    root = np.sqrt(np.square(OTF1_sk0_mag) + np.square(OTF2_mag))
    fact2 = np.zeros((Ny, Nx))
    fact2[ind_mask2] = OTF1_sk0[ind_mask2] / root[ind_mask2]

    val2re = band2re * fact2
    val2im = band2im * fact2
    overlap1 = val2re + 1j * val2im

    overlap0 = F.ifft2(F.ifftshift(overlap0))
    overlap1 = F.ifft2(F.ifftshift(overlap1))

    return overlap0, overlap1


def getmodamp(k0angle, k0length, bands, overlap0, overlap1, Nx, Ny, Nz, order1, order2, dxy, dz, OTF, lamda, redoarrays, pParam):
    k1 = np.zeros(2)
    k1[0] = k0length * np.cos(k0angle)
    k1[1] = k0length * np.sin(k0angle)

    if redoarrays > 0:
        [overlap0, overlap1] = makeoverlaps(bands, Nx, Ny, Nz, order1, order2, k1[0], k1[1], dxy, dz, OTF, lamda)

    dkx = 1 / (Nx * dxy)
    dky = 1 / (Ny * dxy)

    xcent = Nx / 2
    ycent = Ny / 2
    kx = k1[0] * (order2 - order1)
    ky = k1[1] * (order2 - order1)

    ii = np.arange(0, Ny, 1)
    jj = np.arange(0, Nx, 1)
    [jj, ii] = np.meshgrid(jj, ii)
    angle = 2 * np.pi * ((jj - xcent) * (kx / dkx) / Nx + (ii - ycent) * (ky / dky) / Ny)
    expiphi = np.cos(angle) + 1j * np.sin(angle)

    sumXstarY = complex(0, 0)
    sumXmag = 0
    sumYmag = 0

    overlap1_shift = overlap1 * expiphi
    sumXstarY = sumXstarY + np.sum(overlap0.conjugate() * overlap1_shift)
    sumXmag = sumXmag + np.sum(np.abs(np.square(overlap0)))
    sumXmag = sumXmag + np.sum(np.abs(np.square(overlap1)))

    modamp = sumXstarY / sumXmag
    # corr_coef = np.square(np.abs(sumXstarY)) / (sumXmag + sumYmag)
    if pParam.ifshowmodamp == 1:
        print(' In getmodamp: angle=%f, mag=%f 1/micron, amp=%f, phase=%f' % (k0angle, k0length, np.abs(modamp), np.angle(modamp)))
    return modamp


def fitxyparabola(x1, y1, x2, y2, x3, y3):
    if (x1 == x2) or (x2 == x3) or (x3 == x1):
        print('Fit fails; two points are equal: x1=%f, x2=%f, x3=%f\n' % (x1, x2, x3))
        peak = 0
    else:
        xbar1 = 0.5 * (x1 + x2)
        xbar2 = 0.5 * (x2 + x3)
        slope1 = (y2 - y1) / (x2 - x1)
        slope2 = (y3 - y2) / (x3 - x2)
        curve = (slope2 - slope1) / (xbar2 - xbar1)
        if curve == 0:
            print('Fit fails; no curvature: r1=(%f,%f), r2=(%f,%f), r3=(%f,%f) slope1=%f, slope2=%f, curvature=%f\n'% (x1, y1, x2, y2, x3, y3, slope1, slope2, curve))
            peak = 0
        else:
            peak = xbar2 - slope2 / curve
    return peak


def fitk0andmodamps(bands, overlap0, overlap1, Nx, Ny, Nz, k0, dxy, dz, OTF, lamda, pParam):
    # --------------------------------------------------------------------------------
    #                           find optimal k0 and modammps
    # --------------------------------------------------------------------------------
    deltaangle = 0.001
    if Nx >= Ny:
        dkx = 1 / (Nx * dxy)
        deltamag = 0.1 * dkx
    else:
        dky = 1 / (Ny * dxy)
        deltamag = 0.1 * dky

    fitorder1 = 0
    if Nz > 1:
        fitorder2 = 2
    else:
        fitorder2 = 1

    k0mag = np.sqrt(pow(k0[0], 2) + pow(k0[1], 2))
    k0angle = math.atan2(k0[1], k0[0])

    redoarrays = pParam.recalcarrays >= 1

    x2 = k0angle
    modamp = getmodamp(k0angle, k0mag, bands, overlap0, overlap1, Nx, Ny, Nz, fitorder1, fitorder2, dxy, dz, OTF, lamda, redoarrays, pParam)
    amp2 = np.abs(modamp)

    angle = k0angle + deltaangle
    x3 = angle
    modamp = getmodamp(angle, k0mag, bands, overlap0, overlap1, Nx, Ny, Nz, fitorder1, fitorder2, dxy, dz, OTF, lamda, redoarrays, pParam)
    amp3 = np.abs(modamp)

    if amp3 > amp2:
        while amp3 > amp2:
            amp1 = amp2
            x1 = x2
            amp2 = amp3
            x2 = x3
            angle = angle + deltaangle
            x3 = angle
            modamp = getmodamp(angle, k0mag, bands, overlap0, overlap1, Nx, Ny, Nz, fitorder1, fitorder2, dxy, dz, OTF, lamda, redoarrays, pParam)
            amp3 = np.abs(modamp)
    else:
        angle = k0angle
        a = amp3
        amp3 = amp2
        amp2 = a
        a = x3
        x3 = x2
        x2 = a
        while amp3 > amp2:
            amp1 = amp2
            x1 = x2
            amp2 = amp3
            x2 = x3
            angle = angle - deltaangle
            x3 = angle
            modamp = getmodamp(angle, k0mag, bands, overlap0, overlap1, Nx, Ny, Nz, fitorder1, fitorder2, dxy, dz, OTF, lamda, redoarrays, pParam)
            amp3 = np.abs(modamp)
    angle = fitxyparabola(x1, amp1, x2, amp2, x3, amp3)

    x2 = k0mag
    modamp = getmodamp(angle, k0mag, bands, overlap0, overlap1, Nx, Ny, Nz, fitorder1, fitorder2, dxy, dz, OTF, lamda, redoarrays, pParam)
    amp2 = np.abs(modamp)
    mag = k0mag + deltamag
    x3 = mag
    modamp = getmodamp(angle, mag, bands, overlap0, overlap1, Nx, Ny, Nz, fitorder1, fitorder2, dxy, dz, OTF, lamda, redoarrays, pParam)
    amp3 = np.abs(modamp)
    if amp3 > amp2:
        while amp3 > amp2:
            amp1 = amp2
            x1 = x2
            amp2 = amp3
            x2 = x3
            mag = mag + deltamag
            x3 = mag
            modamp = getmodamp(angle, mag, bands, overlap0, overlap1, Nx, Ny, Nz, fitorder1, fitorder2, dxy, dz, OTF, lamda, redoarrays, pParam)
            amp3 = np.abs(modamp)
    else:
        mag = k0mag
        a = amp3
        amp3 = amp2
        amp2 = a
        a = x3
        x3 = x2
        x2 = a
        while amp3 > amp2:
            amp1 = amp2
            x1 = x2
            amp2 = amp3
            x2 = x3
            mag = mag - deltamag
            x3 = mag
            modamp = getmodamp(angle, mag, bands, overlap0, overlap1, Nx, Ny, Nz, fitorder1, fitorder2, dxy, dz, OTF, lamda, redoarrays, pParam)
            amp3 = np.abs(modamp)

    mag = fitxyparabola(x1, amp1, x2, amp2, x3, amp3)

    if pParam.ifshowmodamp == 1:
        print('Optimum modulation amplitude:\n')
    redoarrays = pParam.recalcarrays >= 2
    modamp = getmodamp(angle, mag, bands, overlap0, overlap1, Nx, Ny, Nz, fitorder1, fitorder2, dxy, dz, OTF, lamda, redoarrays, pParam)

    if pParam.ifshowmodamp == 1:
        print('Optimum k0 angle=%f rad, length=%f 1/microns, spacing=%f microns\n'% (angle, mag, 1.0 / mag))

    new_k0 = [mag * np.cos(angle), mag * np.sin(angle)]
    amps = modamp
    return new_k0, amps


def findk0(bands, overlap0, overlap1, Nx, Ny, Nz, k0, dxy, dz, OTF, lamda):

    dkx = 1 / (Nx * dxy)
    dky = 1 / (Ny * dxy)
    fitorder1 = 0
    if Nz > 1:
        fitorder2 = 1
        overlap0, overlap1 = makeoverlaps(bands, Nx, Ny, Nz, fitorder1, fitorder2, k0[0], k0[1], dxy, dz, OTF, lamda)
        crosscorr_c = np.sum(overlap0.conjugate() * overlap1, 3)
    else:
        fitorder2 = 1
        overlap0, overlap1 = makeoverlaps(bands, Nx, Ny, Nz, fitorder1, fitorder2, k0[0], k0[1], dxy, dz, OTF, lamda)
        crosscorr_c = overlap0.conjugate() * overlap1

    crosscorr = F.fftshift(F.ifft2(F.ifftshift(crosscorr_c)))
    index_k0 = np.argmax(np.square(np.abs(crosscorr)))
    new_k0 = np.array([index_k0 % Nx, index_k0 // Nx])
    new_k0[0] = new_k0[0] - Nx / 2
    new_k0[1] = new_k0[1] - Ny / 2
    new_k0 = new_k0 - 1

    if k0[0] / dkx < new_k0[0] - Nx / 2:
        new_k0[0] = new_k0[0] - Nx
    if k0[0] / dkx > new_k0[0] + Nx / 2:
        new_k0[0] = new_k0[0] + Nx
    if k0[1] / dky < new_k0[1] - Ny / 2:
        new_k0[1] = new_k0[1] - Ny
    if k0[1] / dky > new_k0[1] + Ny / 2:
        new_k0[1] = new_k0[1] + Ny

    new_k0 = new_k0 / fitorder2
    new_k0[0] = new_k0[0] * dkx
    new_k0[1] = new_k0[1] * dky

    return new_k0


def cal_modamp(image, OTF, pParam):
    # --------------------------------------------------------------------------------
    #                             parameters initialization
    # --------------------------------------------------------------------------------
    [Nx, Ny] = [pParam.Nx, pParam.Ny]
    lamda = pParam.lamda
    Nx = pParam.Nx
    Ny = pParam.Ny
    space = pParam.space
    dx = pParam.dx
    dy = pParam.dy
    dxy = pParam.dxy
    dkx = pParam.dkx
    dky = pParam.dky
    dkr = pParam.dkr
    nphases = pParam.nphases
    ndirs = pParam.ndirs
    NA = pParam.NA
    norders = pParam.norders
    napodize = pParam.napodize
    k0mod = pParam.k0mod
    k0angle = pParam.k0angle_c
    k0 = np.transpose([k0mod * np.cos(k0angle), k0mod * np.sin(k0angle)])
    global overlap0, overlap1

    # --------------------------------------------------------------------------------
    #                                       read OTF
    # --------------------------------------------------------------------------------
    # headerotf, rawOTF = read_mrc(OTF_path)
    # nxotf = headerotf[0][0]
    # nyotf = headerotf[0][1]
    # dkrotf = headerotf[0][10]
    # diagdist = int(np.sqrt(np.square(Nx / 2) + np.square(Ny / 2)) + 1)
    # k0mag = int(k0mod / dkr)
    # rawOTF = np.squeeze(rawOTF)
    # OTF = rawOTF[0:-1:2]
    # x = np.arange(0, nxotf * dkrotf, dkrotf)
    # xi = np.arange(0, (nxotf-1) * dkrotf, dkr)
    # interp = interp1d(x, OTF, kind='slinear')
    # OTF = interp(xi)
    # sizeOTF = len(OTF)
    # prol_OTF = np.zeros((diagdist + k0mag * (norders - 1) * 4))
    # prol_OTF[0:sizeOTF] = OTF
    # OTF = prol_OTF

    # --------------------------------------------------------------------------------
    #                                calculate modamp
    # --------------------------------------------------------------------------------
    image = np.reshape(image, [Ny, Nx, ndirs, nphases])
    modamp = []
    cur_k0 = k0
    for d in range(ndirs):
        sepMatrix = make_matrix(nphases, norders)
        imagePro = np.squeeze(image[:, :, d, :])
        imagePro = np.transpose(imagePro, (1, 0, 2))
        Fimagepro = []
        for i in range(nphases):
            imagePro[:, :, i] = apodize(napodize, Ny, Nx, 1, imagePro[:, :, i])
            Fimagepro.append(F.fftshift(F.fft2(imagePro[:, :, i])))

        Fimagepro = np.array(Fimagepro).reshape((nphases, Ny * Nx))
        bandsDir = np.dot(sepMatrix, Fimagepro)
        bandsDir = np.reshape(bandsDir, (nphases, Nx, Ny))

        fitorder0 = 0
        fitorder1 = 1
        overlap0, overlap1 = makeoverlaps(bandsDir, Ny, Nx, 1, fitorder0, fitorder1, k0[d, 0], k0[d, 1], dxy, 0, OTF, lamda)
        # k0[d, :] = findk0(bandsDir, overlap0, overlap1, Ny, Nx, 1, k0[d, :], dxy, 0, OTF, lamda)
        new_k0, cur_modamp = fitk0andmodamps(bandsDir, overlap0, overlap1, Ny, Nx, 1, k0[d, :], dxy, 0, OTF, lamda, pParam)
        cur_k0[d, :] = new_k0

        modamp.append(cur_modamp)

    return cur_k0, modamp

