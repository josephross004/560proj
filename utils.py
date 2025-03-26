from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
import numpy as np
from PIL import Image

def wavread(path: str, stereo=True):
    '''
    reads in a wav file. 
    input: a string to the path of the wave file.
    output: a tuple, where the 0th item is the sample rate and the 1st item is a vector of the waveform.
    '''
    samplerate, data = wavfile.read(path)

    #NOTE: The files in the dataset are usually in stereo format (2 channels) and need to be averaged.
    #      This is a flag you can set to bypass this. 

    if(stereo):
        data = (data[:,0]+data[:,1])/2
    return samplerate, data

def saveAsVector(data: np.ndarray, path: str):
    '''
    saves a vector as a numpy array in a separate file.
    input: data vector, output path.
    output: none
    '''
    np.save(path+'.npy', data)

def normalizeMatrix(matrix: np.ndarray):
    '''
    converts a matrix to a normalized matrix (all values are [0,1]). 
    input: matrix
    output: matrix
    '''
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

def saveAsSpectrogram(data: np.ndarray, rate: int, path: str):
    '''
    saves a vector as a spectrogram image in a separate file.
    input: data vector, sampling rate (Hz), output path.
    output: none
    '''
    g_std = 12
    print(data)
    print(data.shape)
    window = gaussian(20, std=g_std, sym=True)
    mfft = rate
    stft = ShortTimeFFT(window, hop=50, fs=rate, mfft=mfft, scale_to='psd')
    print("fft created.")
    Sxx = stft.spectrogram(np.transpose(data))
    Sxx = 10 * np.log10(Sxx) #convert to dB
    print("spectro done.")
    matrix = (normalizeMatrix(Sxx)*255).astype(np.uint8)
    print(f"matrix {matrix} of size {matrix.shape}")
    image = Image.fromarray(matrix, mode="L")
    image.save(path+".png")



