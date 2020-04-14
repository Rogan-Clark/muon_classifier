import numpy as np
from numpy.random import RandomState
import zern_core as zern
import matplotlib.pyplot as plt

def add_noise(picturearr,std_dev,batch_size):
    '''Adds random noise to cropped CHEC camera images.'''
    noisearr = np.random.normal(0,std_dev,batch_size*48*48)
    noisearr = noisearr.reshape((batch_size,48,48,1))
    for b in range(batch_size):
           for i in range(0,8):
               for j in range(0,8):
                   noisearr[b][i][j][0]=0
                   noisearr[b][47-i][j][0]=0
                   noisearr[b][i][47-j][0]=0
                   noisearr[b][47-i][47-j][0]=0
    pictures = picturearr.copy() + noisearr * np.nanmean(picturearr) / 10.0
    print(np.mean(picturearr))
    return pictures

def add_star(picturearr,batch_size):
    '''Simulates a random star being added to every field in a cropped CHEC picture'''
    pictures = picturearr.copy()
    picrange = np.arange(batch_size)
    for k in picrange:            
        dumvar=0
        while dumvar==0:
            rand1 = np.random.randint(0,48)
            rand2 = np.random.randint(0,48)
            if rand1<8 or rand1>=40:
                if rand2<8 or rand2>=40:
                    pass
                else:
                    dumvar=1
            else:
                dumvar=1
        pictures[k,rand1,rand2,0] = pictures[k,rand1,rand2,0] + np.random.uniform(0,1) * np.amax(pictures[k,:,:,0])

    return pictures

def add_bad_flatfield(picturearr,batch_size):
    '''Simulates a set of bad flat field calibrations for a CHEC field using Alvaro's ZERN code.'''
    pictures = picturearr.copy()
    picrange = np.arange(batch_size)
    zerns = rand_zern(np.random.randint(100))
    #zerns = (zerns + 1.0)/2.0 #Normalize to 0,1
    zerns = (zerns) + 1.0
    #plt.imshow(zerns)
    #plt.colorbar()
    #plt.show()
    for m in picrange:
        #plt.imshow(zerns)
        #plt.colorbar()
        #plt.savefig('zern.png')
        #raise KeyboardInterrupt
        pictures[m,:,:,0] = (pictures[m,:,:,0]+pictures[m,:,:,0]*zerns/10.0)
    pictures = pictures
    return pictures

def rand_zern(randker):
    N = 48
    N_zern = 10
    rho_max = 0.9
    eps_rho = 1.4
    randgen = RandomState(randker)
    extents = [-1, 1, -1, 1]

    # Construct the coordinates
    x = np.linspace(-rho_max, rho_max, N)
    rho_spacing = x[1] - x[0]
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(xx, yy)
    aperture_mask = rho <= eps_rho * rho_max
    rho, theta = rho[aperture_mask], theta[aperture_mask]
    rho_max = np.max(rho)
    extends = [-rho_max, rho_max, -rho_max, rho_max]

    # Compute the Zernike series
    coef = randgen.normal(size=N_zern)
    z = zern.ZernikeNaive(mask=aperture_mask)
    phase_map = z(coef=coef, rho=rho, theta=theta, normalize_noll=False, mode='Jacobi', print_option='Silent')
    phase_map = zern.rescale_phase_map(phase_map, peak=1)
    phase_2d = zern.invert_mask(phase_map, aperture_mask)

    return phase_2d

if __name__ == "__main__":
    phase=rand_zern(np.random.randint(100))
    plt.imshow(phase)
    plt.colorbar()
    plt.show()
