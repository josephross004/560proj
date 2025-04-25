'''
Instructions: 
Make sure that Animal-Sound-Dataset submodule is fully init'd 
and all of the .wav files are available. 

The output folder "./pdata" is going to expand when this program is run. 
The structure will look like this: 

pdata/
    spectrograms/
        training/
        testing/
        validation/
    spectra/
        training/
        testing/
        validation/
    waveforms/
        training/
        testing/
        validation/

NB: For testing, run this program in the command line like so: 

python prepareData.py

NB: For real data preparation/expansion, run this program in the command line like so:

python prepareData.py -f
'''

import os
import sys
import utils
import tqdm
import warnings

TEST = True

try:
    if (len(sys.argv) != 2 or sys.argv[1] != '-f'):
        print("Note: this program is running in TEST mode.")
    else:
        print("Note: this program is running in REAL mode.")
        TEST = False
except IndexError: 
    print("Note: this program is running in TEST mode.")

def find_files(directory):
    file_paths = []
    count = 0
    filesize = 0
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_paths.append(file_path)
            filesize += os.path.getsize(file_path)
            count+=1
    return file_paths, count, filesize

def order_file_size(filesize):
    if filesize < 1000:
        return str(filesize) + " B"
    elif filesize < 1000000:
        return str(filesize // 1000) + " kB"
    elif filesize < 1000000000:
        return str(filesize // 1000000) + " MB"    
    elif filesize < 1000000000000:
        return str(filesize // 1000000000000) + " GB"

#create a manifest of files to go through
print("creating manifest...")
list, count, filesize = find_files("./Animal-Sound-Dataset/")
list = list[2:]

print(f"found {count-2} files, total size = {order_file_size(filesize)} ")

if TEST:
    list = list[0:9]


reqd_folders = [
"./pdata",
"./pdata/spectrograms",
"./pdata/spectrograms/testing",
"./pdata/spectrograms/training",
"./pdata/spectrograms/validation",
"./pdata/waveforms",
"./pdata/waveforms/testing",
"./pdata/waveforms/training",
"./pdata/waveforms/validation",
"./pdata/spectra",
"./pdata/spectra/testing",
"./pdata/spectra/training",
"./pdata/spectra/validation"]


for r in reqd_folders:
    if not os.path.exists(r):
        os.makedirs(r)
#TODO: spectra in utils.

with open("./pdata/manifest.txt","w") as f:
    for i in list:
        f.write(i)
        f.write("\n")
print("manifest saved at ./pdata/manifest.txt")


import random
warnings.filterwarnings('error',category=RuntimeWarning)


with tqdm.tqdm(list) as pbar:
    a = [0,1,2,3,4,5,6,7,8,9]
    for i in pbar:
        ttv = random.choice(a)
        a.remove(ttv)
        pbar.set_description(f"sending to {int(ttv)}: {os.path.basename(i)}")
        try:
            sd = utils.SoundData(i)
        except ValueError or RuntimeWarning:
            import subprocess
            if not os.path.exists("./converted"):
                os.makedirs("./converted")
            subprocess.call(['ffmpeg', '-i', i,'./converted/'+str(os.path.basename(i)), '-loglevel','quiet', '-y'])
            i = "./converted/"+str(os.path.basename(i))
        if (ttv) < 1:
            sd.saveAsSpectrogram("./pdata/spectrograms/validation/"+os.path.basename(i))
            sd.saveAsVector("./pdata/waveforms/validation/"+os.path.basename(i))
            sd.saveAsSpectra("./pdata/spectra/validation/"+os.path.basename(i))
        elif (ttv) < 2:
            sd.saveAsSpectrogram("./pdata/spectrograms/testing/"+os.path.basename(i))
            sd.saveAsVector("./pdata/waveforms/testing/"+os.path.basename(i))
            sd.saveAsSpectra("./pdata/spectra/testing/"+os.path.basename(i))
        else:
            sd.saveAsSpectrogram("./pdata/spectrograms/training/"+os.path.basename(i))
            sd.saveAsVector("./pdata/waveforms/training/"+os.path.basename(i))
            sd.saveAsSpectra("./pdata/spectra/training/"+os.path.basename(i))
        if len(a)==0:
            a = [0,1,2,3,4,5,6,7,8,9]
