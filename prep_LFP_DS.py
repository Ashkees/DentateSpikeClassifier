import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import neo
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
from getpass import getuser
import pickle

all_paths = {'ashley': {'codepath': Path(r'E:\Mulle\Syt7'),
                        'datapath': Path(r'E:\Mulle\Ashley_data\Data\2021_Syt7Flox_awake_DG_CA3')}}

upaths = all_paths[getuser()]
upaths['intan_scripts'] = upaths['codepath'] / 'load_intan_rhd_format'
upaths['data_to_analyze'] = upaths['codepath'] / 'Data_to_Analyze_Syt7Flox.xlsx'
upaths['output'] = upaths['codepath'] / 'DS_LFPs'
upaths['processed_DS'] = upaths['codepath'] / 'dataset' / 'dentate_spikes'

sheet_name = 'Syt7FLox_DGlfpML'
excel_col = 'largest_DS'
# parameters for broadband filtering and downsampling
ds_factor = 16
highpass = 0.2
lowpass = 300


def ds(trace, ds_factor):
    signal_ds = np.mean(np.resize(trace,
                        (int(np.floor(trace.size/ds_factor)), ds_factor)), 1)
    return signal_ds

def load_data(data_folder, ch_order, fs, ds_factor, lowpass, highpass):

    intan_file = data_folder / 'amplifier.dat'

    fs = 20000
    nb_channel = 96

    # load the correct channel from the intan file


    r = neo.io.RawBinarySignalIO(intan_file, dtype='int16', sampling_rate=fs,
                                 nb_channel=nb_channel, signal_gain=0.195)
    seg = r.read_segment(lazy=True, signal_group_mode='group-by-same-units')
    filt_LFP = np.zeros([int(np.floor(seg.analogsignals[0].shape[0] / ds_factor)), len(ch_order)])
    nyq = fs / (2 * ds_factor)
    b_hi, a_hi = signal.butter(4, highpass / nyq, "high", analog=False)
    b_lo, a_lo = signal.butter(4, lowpass / nyq, "low", analog=False)
    for ch_ix, c_ch in enumerate(ch_order):
        anasig_raw = seg.analogsignals[0].load(channel_indexes=[c_ch])
        # extract just the plain np array from the neo object
        hf_LFP = anasig_raw.magnitude
        LFP_ts = anasig_raw.times
        ds_LFP = ds(hf_LFP, ds_factor)
        LFP_highpass = signal.filtfilt(b_hi, a_hi, ds_LFP)
        filt_LFP[:, ch_ix] = signal.filtfilt(b_lo, a_lo, LFP_highpass)
    ds_ts = ds(LFP_ts, ds_factor)

    return ds_ts, filt_LFP

def filter_data(traces, ts, fs, ds_factor, lowpass, highpass):

    nyq = fs / (2 * ds_factor)
    b_hi, a_hi = signal.butter(4, highpass / nyq, "high", analog=False)
    b_lo, a_lo = signal.butter(4, lowpass / nyq, "low", analog=False)
    filt_LFP = np.zeros([int(np.floor(traces.shape[0] / ds_factor)), traces.shape[1]])
    for j in np.arange(traces.shape[1]):  # take only 3 out of the 8 channels
        ds_LFP = ds(traces[:, j], ds_factor)
        LFP_highpass = signal.filtfilt(b_hi, a_hi, ds_LFP)
        filt_LFP[:, j] = signal.filtfilt(b_lo, a_lo, LFP_highpass)
    ds_ts = ds(ts, ds_factor)

    return filt_LFP, ds_ts





if __name__ == '__main__':

    # load the excel sheet list of data
    data_list = pd.read_excel(upaths['data_to_analyze'], sheet_name=sheet_name)
    # list the channels in order
    ch = [64, 95, 65, 94, 68, 91, 69, 90, 79, 80, 77, 89, 78, 81, 66, 88, 76, 82, 67, 87, 75, 83, 73,
          86, 74, 93, 71, 85, 72, 84, 70, 92]

    for i in tqdm(np.arange(1, len(data_list))):
        if ~np.isnan(data_list[excel_col][i]):

            # get the intan file name
            date_folder = '20' + data_list['Filename'][i][3:9]
            intan_folder = data_list['Filename'][i]
            intan_path = upaths['datapath'] / date_folder / intan_folder

            ts, lfp = load_data(intan_path, ch, 20000, ds_factor, lowpass, highpass)
            #lfp, ts = filter_data(hf_lfp, hf_ts, fs=20000, ds_factor=ds_factor, lowpass=lowpass, highpass=highpass)

            # put the 3 channels of downsampled LFP into the data list to save later
            c_data = {}
            c_data['date'] = intan_folder
            c_data['LFP_channels'] = ch
            c_data['DS_ch_num'] = data_list[excel_col][i]
            c_data['DS_ch_index'] = ch.index(data_list[excel_col][i])
            c_data['fs'] = int(20000 / ds_factor)
            c_data['lfp'] = lfp
            c_data['lfp_ts'] = ts

            # save the pickle for that session
            pickle_file = upaths['output'] / f'{intan_folder}.pkl'
            with open(pickle_file, 'wb') as fp:
                pickle.dump(c_data, fp)
