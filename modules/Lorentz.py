import os
import sys
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

def lorentzian_fw(x_pos,
                  y_shift,
                  x_center,
                  amplitude,
                  full_width):
    return y_shift + (2*amplitude/np.pi) * (full_width / (4*(x_pos-x_center)**2 + full_width**2))

def lorentzian(x_pos,
               x_center,
               amplitude,
               half_width):
    return amplitude/(1+((x_pos-x_center)/half_width)**2)

class lorentz:
    def __init__(self, data, params):
        print('Lorentz fitting for SED...', flush=True)
        self.q_ind = params.q_slice_index
        self.sed = data.sed_avg[:,self.q_ind]
        
        self._peak_num_guesses = params.num_guesses
        self._bounds = np.zeros((self._peak_num_guesses, 2))
        
        # some bounds on the fitting. Might need to tweak these
        self.dx = 10
        self.dxarr = 100
        self.maxfev = 5e3
        
        self.xarr = np.arange(len(self.sed))
        self.popt = np.zeros((self._peak_num_guesses, 3))
        self.pcov = np.zeros((self._peak_num_guesses, 3))

        # for auto-fitting
        self._fit_auto = bool(params.lorentz_fit_mode == 'auto')
        self._out_prefix = params.out_prefix
        self._fit_results_folder = self._out_prefix + '_Lorentz_Fitting'
        self._fit_results_save_path = os.path.join(params.work_path, self._fit_results_folder)
        
        self._sed_fit_auto = data.sed_avg
        self._qpoints_fit_auto = data.qpoints
        self._thz_fit_auto = data.thz
    
    def fit(self):
        if not self._fit_auto: 
            for i in range(self._peak_num_guesses):
                if self._peak_num_guesses[i]-self.dxarr < 0:
                    start = 0
                else:
                    start = self._peak_num_guesses[i]-self.dxarr
                if self._peak_num_guesses[i]+self.dxarr > len(self.sed)-1:
                    end = len(self.sed)-1
                else:
                    end = self._peak_num_guesses[i]+self.dxarr
                self._bounds[i, 0] = start
                self._bounds[i, 1] = end
                lb = [self._peak_num_guesses[i] - self.dx, 1e-6, 1e-10]
                ub = [self._peak_num_guesses[i] + self.dx, np.inf, 1e3]

                try:
                    self.popt[i,:], pcov = curve_fit(
                        lorentzian,
                        self.xarr[start:end],
                        self.sed[start:end],
                        p0=[
                            self._peak_num_guesses[i],
                            self.sed[self._peak_num_guesses[i]],
                            1
                        ],
                        bounds= (lb,ub),
                        maxfev= self.maxfev,
                    )
                    self.pcov[i,:] = np.sqrt(np.diag(pcov))
                except:
                    print('\nWARNING: Lorentz fit for peak-guess-index {} failed!\n'
                            .format(self._peak_num_guesses[i]),flush=True)
                    continue
        else:
            self.fit_auto()
            
    def fit_auto(self):
        '''
        Here I define my own Lorentz fitting function to enable 
        the automatic lorentz fitting and lifetime calculation 
        '''
        peak_distance_guess = 10   # peaks minmized distance
        peak_height_guess   = 1e-5 # peaks minmized height
        peak_prominence_guess = peak_height_guess*0.8 #5e-6 # peaks prominence
        half_width_guess = 1e-2 # max half_width at the half-height of peak

        num_q_points = self._qpoints_fit_auto.shape[0]
        lifetime_file  = self._out_prefix + '_fitted_PhononLifetime.txt'
        q_point_lifetime_file = None

        y_shift  = 1e-12
        x_center = 0
        peak_amplitude  = peak_height_guess*0.1
        peak_half_width = half_width_guess
        peak_full_width = peak_half_width*2

        if not os.path.exists(self._fit_results_save_path):
            os.mkdir(self._fit_results_save_path)

        with open(os.path.join(self._fit_results_save_path, lifetime_file), 'w') as fw:
                fw.write('# iq  Frequency(THz)  Half_width(THz)  Lifetime(ps)\n')
        
        for iq in range(num_q_points):
            print('\nFitting Q-point {:d}...'.format(iq), flush=True)
            y_shift = np.min(self._sed_fit_auto[:, iq])
            q_point_lifetime_file = self._out_prefix + '_fitted_PhononLifetime_qpoint{:d}.txt'.format(iq)
            with open(os.path.join(self._fit_results_save_path, q_point_lifetime_file), 'w') as fw:
                    fw.write('# iq  Frequency(THz)  Half_width(THz)  Lifetime(ps)\n')
            
            # Find peak centers on frequency axis
            peaks, amp_max = find_peaks(
                self._sed_fit_auto[:, iq], 
                distance = peak_distance_guess, 
                height = peak_height_guess, 
                prominence = peak_prominence_guess
            )
            print('Found {:d} bands on Q-point {:d}'.format(len(peaks), iq), flush=True)
            
            # Fit lorentz curve
            for peak_center in peaks:
                inital_guess = (
                    y_shift,
                    self._thz_fit_auto[peak_center], 
                    self._sed_fit_auto[peak_center, iq], 
                    half_width_guess,
                )
                # print(peak_center, inital_guess, self._thz_fit_auto[peak_center], type(self._thz_fit_auto[peak_center]), self._sed_fit_auto[peak_center, iq], type(self._sed_fit_auto[peak_center, iq]))
                # print(self._thz_fit_auto)
                # print(self._sed_fit_auto[:, iq])
                popt, pcov = curve_fit(
                    lorentzian_fw, 
                    self._thz_fit_auto, 
                    self._sed_fit_auto[:, iq], 
                    p0 = inital_guess, 
                    absolute_sigma=True, 
                    maxfev=int(self.maxfev),
                )
                # x_center, peak_amplitude, peak_half_width = popt
                # peak_full_width = peak_half_width*2

                y_shift, x_center, peak_amplitude, peak_full_width = popt
                
                with open(os.path.join(self._fit_results_save_path, q_point_lifetime_file), 'a') as fa:
                    fa.write('{:3d} {:13.8f} {:13.8f} {:13.8f}\n'.format(iq, x_center, peak_full_width, abs(1/2/peak_full_width/np.pi)))
                
                with open(os.path.join(self._fit_results_save_path, lifetime_file), 'a') as fa:
                    fa.write('{:3d} {:13.8f} {:13.8f} {:13.8f}\n'.format(iq, x_center, peak_full_width, abs(1/2/peak_full_width/np.pi)))
            
            print('Q-point {:d} fitting done'.format(iq), flush=True)                
        

