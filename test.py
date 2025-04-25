import os
import utils
import numpy as np

print(f"\n \n \n PATH: {os.getcwd()} \n \n \n ")

#test 1: see if you can read in a wav file.
wv = utils.SoundData("/work/users/r/o/ross004/560/Animal-Sound-Dataset/Aslan/aslan_1.wav")

#test 2: see if you can save a vector as a npy file.
#wv.saveAsVector("./testout/wav")

#test 3: see if you can save a vector as a spectrogram.
wv.saveAsSpectrogram("./testout/img")

#test 4: see if you can save a vector as spectra.
wv.saveAsSpectra("./testout/spectra")