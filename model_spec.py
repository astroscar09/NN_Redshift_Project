import numpy as np
from scipy.optimize import curve_fit

class ModelSpectrum:
    def __init__(self, wavelengths):
        self.wavelengths = wavelengths
        self.gaussians = []
        self.continuum_params = []

    def add_gaussian(self, amplitude, mean, stddev):
        self.gaussians.append((amplitude, mean, stddev))

    def set_continuum(self, *params):
        self.continuum_params = params

    def gaussian(self, x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

    def continuum(self, x, *params):
        # Example: polynomial continuum
        return sum(p * x**i for i, p in enumerate(params))

    def model(self, x):
        y = self.continuum(x, *self.continuum_params)
        for amplitude, mean, stddev in self.gaussians:
            y += self.gaussian(x, amplitude, mean, stddev)
        return y
    
    def add_noise(self, flux_error):
        return self.model(self.wavelengths) + np.random.normal(0, flux_error, len(self.wavelengths))


#Example usage:
# wavelengths = np.linspace(4000, 7000, 1000)
# model = ModelSpectrum(wavelengths)
# model.add_gaussian(1.0, 5000, 50)
# model.set_continuum(1.0, -0.001, 0.000001)
# spectrum = model.model(wavelengths)

def make_model(noise):
    
    x = np.linspace(900, 11000, 5000)
    
    model = ModelSpectrum(x)
    
    #NV
    model.add_gaussian(0.5e-19, 1238.821, 3)
    model.add_gaussian(0.5e-19, 1242.804, 3)

    #NIV
    model.add_gaussian(0.5e-19, 1486.496, 3)

    #CIV
    model.add_gaussian(0.5e-19, 1548.187, 3)
    model.add_gaussian(0.5e-19, 1550.772, 3)

    #CIII
    model.add_gaussian(0.5e-19, 1908.734, 3)

    #MgII
    model.add_gaussian(0.5e-19, 2795.528, 3)
    model.add_gaussian(0.5e-19, 2802.705, 3)
    
    #OII
    model.add_gaussian(0.5e-19, 3727, 3)
    model.add_gaussian(0.5e-19, 3729, 3)
    
    #HBeta
    model.add_gaussian(1e-19, 4864, 10)
    
    #OIII
    model.add_gaussian(3e-19, 4960, 11)
    model.add_gaussian(9e-19, 5007, 11)
    
    #Halpha
    model.add_gaussian(5e-19, 6564, 10)
    
    #Continuum set to 0 to start off
    model.set_continuum(0, 0, 0)
    
    spec = model.add_noise(noise)

    return x, spec