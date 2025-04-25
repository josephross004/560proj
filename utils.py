from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
import numpy as np
from PIL import Image
import parselmouth
import matplotlib.pyplot as plt
import warnings
class SoundData:
    def __init__(self, path: str, stereo=True):
        '''
        reads in a wav file. 
        input: a string to the path of the wave file.
        output: a tuple, where the 0th item is the sample rate and the 1st item is a vector of the waveform.
        '''

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=wavfile.WavFileWarning)
            self.samplerate, self.data = wavfile.read(path)
            

        #NOTE: The files in the dataset are usually in stereo format (2 channels) and need to be averaged.
        #      This is a flag you can set to bypass this. 

        if(self.data.ndim == 2):
            self.data = (self.data[:,0]+self.data[:,1])/2

        self.p_sound = parselmouth.Sound(path)
        

    def saveAsVector(self, path: str):
        '''
        saves a vector as a numpy array in a separate file.
        input: data vector, output path.
        output: none
        '''
        np.save(path+'.npy', self.data)

    def normalizeMatrix(self, matrix: np.ndarray):
        '''
        converts a matrix to a normalized one (all values are [0,1]). 
        input: matrix
        output: matrix
        '''
        matrix /= np.max(np.abs(matrix),axis=0)
        matrix = abs(matrix-255)
        return matrix


    def saveAsSpectrogram(self, path: str, dynamic_range=70):
        '''
        saves a vector as a spectrogram image in a separate file.
        input: data vector, sampling rate (Hz), output path.
        output: none
        '''
        spectrogram = self.p_sound.to_spectrogram(window_length=0.025, maximum_frequency=self.samplerate/2)
        fig = plt.figure(frameon=False)
        X, Y = spectrogram.x_grid(), spectrogram.y_grid()
        spectrogram.values[spectrogram.values == 0] = np.min(spectrogram.values[np.nonzero(spectrogram.values)])
        sg_db = 10 * np.log10(spectrogram.values)
        plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='Greys')
        plt.ylim([spectrogram.ymin, spectrogram.ymax])
        plt.axis('off')
        fig.savefig(path+".png", bbox_inches='tight', pad_inches=0)
        plt.close()

    def saveAsSpectra(self, path: str):
        '''
        saves a vector as a spectral matrix. 
        input: data vector, sampling rate, output path. 
        output: none
        '''
        pctls = [1,5,10,25,50,75,90,95,99]
        spectrogram = self.p_sound.to_spectrogram(frequency_step=10.0, window_length=0.025, maximum_frequency=self.samplerate/2)
        out = np.zeros((len(pctls), (spectrogram.values.shape[0])))
        spectrogram.values[spectrogram.values == 0] = np.min(spectrogram.values[np.nonzero(spectrogram.values)])
        sg_db = (10 * np.log10(spectrogram.values))
        sgn_db = sg_db + 35
        for i in range(spectrogram.values.shape[0]):
            for j in range(len(pctls)):
                out[j][i] = np.percentile(sgn_db[i], pctls[j])
        idx = np.argwhere(np.all(out[..., :] == 0, axis=0))
        out = np.delete(out,idx,axis=1)
        np.save(path+".npy", out)
                



    

