
import librosa
import os
import json


DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # One second worth of sound

# hop_length - segment size for each snapshot
# num_fft - window size for FFT

def prepareDataset(dataset_path, json_path, num_mfcc = 13,hop_length = 512, num_fft = 2048):
    # data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs" : [],
        "files" : []
    }
    # loop through all the sub dirs
    # walk - enables to walk through folder structure recursively
    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):
        # we need to ensure that we're not at a root level
        if dirpath is dataset_path:
            continue
        # update the mappings
        category = dirpath.split("/")[-1] # dataset/down --> [dataset,down] [-1]--> [down]
        data["mappings"].append(category)
        print(f"Processing {category}")
        
        # loop through all the filenames and extract MFCCs
        for f in filenames:
            # get file path (reconstruct)
            file_path = os.path.join(dirpath,f);
            # load audio file
            signal,sample_rate = librosa.load(file_path)
            # ensure the audio file is at least one second
            if len(signal) >= SAMPLES_TO_CONSIDER:
                # enforce 1 second long signal
                signal = signal[:SAMPLES_TO_CONSIDER]
                
                # extract the MFCCs
                MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=num_fft,
                                                 hop_length=hop_length)
                
                # store data
                data["labels"].append(i-1) # i (from uppermost loop) is the folder we are in in dataset, 0 is dataset, 1 is down and so forth, we dont want to start at dataset so thats why we subtract 1 from i
                data["MFCCs"].append(MFCCs.T.tolist()) # cast numpy array to python list
                data["files"].append(file_path)
                print("{}: {}".format(file_path, i-1))
    # store in json file
    with open(json_path,"w") as fp:
        json.dump(data, fp, indent=4)
        
if __name__ == "__main__":
    prepareDataset(DATASET_PATH,JSON_PATH)