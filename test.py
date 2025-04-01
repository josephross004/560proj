import utils

#test 1: see if you can read in a wav file.
wv = utils.wavread("./Animal-Sound-Dataset/Aslan/aslan_1.wav")

#test 2: see if you can save a vector as a npy file.
utils.saveAsVector(wv[1], "./testout/wav")

#test 3: see if you can save a vector as a spectrogram.
utils.saveAsSpectrogram(wv[1], wv[0], "./testout/img")