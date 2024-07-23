import pickle
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

from prep_LFP_DS import upaths, sheet_name

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
viridis32 = mpl.colormaps['viridis_r'].resampled(32).colors

# Suppress the specific warning due to sklearn (model was saved in previous sklean version)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

## import custom libraries
import utils
import dSpikes_model as dSpikesClassifier




def load_one(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_one_lfp(lfp, index):
    return lfp[:, index]


def calc_csd(lfp, delta_x):
    nChs = np.size(lfp, 0)
    nSamples = np.size(lfp, 1)

    CSD = np.zeros((nChs, nSamples))
    for chi in range(1, nChs - 1):
        CSD[chi, :] = -(lfp[chi - 1, :] - 2 * lfp[chi, :] + lfp[chi + 1, :])

    return CSD


def plot_snip_lfp(ts, snip, dg_ix, ref_ix, cmap=viridis32):
    fig, ax = plt.subplots(1, 1, figsize=[14, 12])
    offset = 100
    for ch in np.arange(snip.shape[0]):
        lw = 2
        zorder = 2
        if ch == dg_ix:
            color = 'xkcd:magenta'
        elif ch == ref_ix:
            color = 'black'
        else:
            color = cmap[ch]
            lw = 1
            zorder = 1
        ax.plot(ts, snip[ch, :] - ch*offset, c=color, lw=lw, zorder=zorder)
    ax.axvline(0, ls='--')
    return ax


def plot_snip_csd(ts, lfp_snip, csd_snip, cmap='coolwarm'):
    fig, ax = plt.subplots(1, 1, figsize=[14, 12])
    ax.imshow(csd_snip, cmap=cmap, vmax=0.1, vmin=-0.1)
    offset = 100
    for ch in np.arange(lfp_snip.shape[0]):
        ax.plot(ts, lfp_snip[ch, :] - ch * offset, c='black', zorder=2)


def plot_all_snips(d_dict, fig_folder):
    fig_folder.mkdir(exist_ok=True, parents=True)
    for j in np.arange(d_dict['ds_snips'].shape[0]):
        ax = plot_snip_lfp(data['ts_snips'], data['ds_snips'][j, :, :], data['DS_ch_index'], data['ref_ch_index'])
        plt.savefig(fig_folder / f'{j}.png')
        plt.close()
    ax = plot_snip_lfp(data['ts_snips'], np.mean(data['ds_snips'], axis=0), data['DS_ch_index'], data['ref_ch_index'])
    plt.savefig(fig_folder / 'avg.png')
    plt.close()


def plot_all_csd(d_dict, fig_folder):
    fig_folder.mkdir(exist_ok=True, parents=True)
    # Set color levels for CSD plot
    levels_ = np.linspace(-200, 200, 50)
    layers = {'ref ch': d_dict['ref_ch_index'], 'DS ch': d_dict['DS_ch_index']}
    for j in np.arange(d_dict['ds_snips'].shape[0]):
        fig, ax = plt.subplots(1, 1, figsize=[14, 12])
        csd_ds = utils.runCSD(data['ds_snips'][j, :, :], spacing=50)
        utils.plotCSD(csd_ds, data['ds_snips'][j, :, :], data['ts_snips'], spacing=1,
                      levels_=levels_,
                      xlim=None,
                      cmap='seismic',
                      ax_lbl={'ylabel': 'Layer', 'xlabel': 'Time from dentate spike (ms)'},
                      ax=ax)
        utils.probe_yticks([])
        utils.probe_yticks(layers)
        plt.xlim([-100, 100])
        plt.tight_layout()
        plt.savefig(fig_folder / f'{j}_csd.png')
        plt.close()
    # plot the average
    fig, ax = plt.subplots(1, 1, figsize=[14, 12])
    avg_lfp = np.mean(data['ds_snips'], axis=0)
    avg_csd = utils.runCSD(avg_lfp, spacing=50)
    utils.plotCSD(avg_csd, avg_lfp, data['ts_snips'], spacing=1,
                  levels_=levels_,
                  xlim=None,
                  cmap='seismic',
                  ax_lbl={'ylabel': 'Layer', 'xlabel': 'Time from dentate spike (ms)'},
                  ax=ax)
    utils.probe_yticks([])
    utils.probe_yticks(layers)
    plt.xlim([-100, 100])
    plt.tight_layout()
    plt.savefig(fig_folder / 'avg_csd.png')
    plt.close()


def plot_local_minima(csd_t0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
    local_min = []
    for j in np.arange(csd_t0.shape[0]):
        local_min.append(utils.detect_peaks(-1 * csd_t0[j, :], mph=0, threshold=0))
    ax[0].plot(csd_t0.T)
    local_min = np.concatenate(local_min)
    ax[1].hist(local_min, bins=np.arange(csd_t0.shape[1]+1))


def match_local_minima(csd_t0, ml_ind):
    matching_min = []
    for j in np.arange(csd_t0.shape[0]):
        local_min = utils.detect_peaks(-1 * csd_t0[j, :], mph=0, threshold=0)
        matching_min.append(tuple(m for m in local_min if m in ml_ind))
    counts = Counter(tuple(e) for e in matching_min)
    return matching_min, counts


def plot_all_local_min(d_dict, fig_folder):
    fig_folder.mkdir(exist_ok=True, parents=True)
    ts0 = np.where(d_dict['ts_snips'] == 0)
    csd_t0 = np.squeeze(d_dict['csd_snips'][:, :, ts0])
    fig, ax = plt.subplots(2, 1, sharex=True)
    plot_local_minima(csd_t0, ax=ax)
    ax[1].axvline(d_dict['DS_ch_index'])
    plt.tight_layout()
    date = d_dict['date']
    plt.savefig(fig_folder / f'{date}.png')
    plt.close()

def detect_ds(d_dict, ref_ch_offset=None):
    if ref_ch_offset is not None:
        ref_chi = data['DS_ch_index'] - ref_ch_offset
    else:
        ref_chi = 0
    d_dict['ref_ch_index'] = ref_chi
    lfp_dg = get_one_lfp(d_dict['lfp'], d_dict['DS_ch_index'])
    lfp_ref = get_one_lfp(d_dict['lfp'], ref_chi)
    lfps = np.row_stack((lfp_ref, lfp_dg))
    sr = d_dict['fs']
    dspikes = dSpikesClassifier.runDSpikes_detection(lfps, dg_chi=1)
    return dspikes



go_on = True
SHOW = False

if __name__ == '__main__':
    data_folder = Path(r'E:\Mulle\Syt7\DS_LFPs')
    file_list = list(data_folder.rglob('*.pkl'))
    data_list = pd.read_excel(upaths['data_to_analyze'], sheet_name=sheet_name)
    ml_index = data_list['CS_min_ind']
    # layers = np.load(r'E:\Mulle\DentateSpikeClassifier\Data\mvl33-221109.probe.layers', allow_pickle=True)

if go_on:

    all_data = []

    for i in tqdm(np.arange(len(file_list))):
        file = file_list[i]
        data = load_one(file)
        dspikes = detect_ds(data)
        data['dspikes'] = data['lfp_ts'][dspikes]
        sr = data['fs']
        if dspikes.shape[0] > 5:
            data['ml_index'] = [int(f) for f in ml_index[i].split(',')]
            ## Trigger LFP in all channels with dentate spikes times
            win_trigger_ms = 200  # window size in ms used to trigger lfp for each event
            win_trigger_samples = int((win_trigger_ms / 1000) * sr)  # convert to samples
            lfp_dspikes, taxis = utils.triggeredAverage(data['lfp'].T, dspikes, taLen=win_trigger_samples, average=False)
            data['ds_snips'] = lfp_dspikes
            data['ts_snips'] = taxis
            # calculate the csd for each snip
            csd = np.zeros_like(lfp_dspikes)
            for j in np.arange(lfp_dspikes.shape[0]):
                csd[j, :, :] = utils.runCSD(lfp_dspikes[j, :, :], smooth=True, spacing=50)  # units are volts and meters?
            data['csd_snips'] = csd
            del data['lfp']
            del data['lfp_ts']
            # save the pickle for that sessio
            date = data['date']
            pickle_file = upaths['processed_DS'] / f'{date}.pkl'
            with open(pickle_file, 'wb') as fp:
                pickle.dump(data, fp)
            all_data.append(data)

            if SHOW:
                fig_folder = Path(r'E:\Mulle\Syt7\Figures\dentate_spikes\ds_detection\ref_chan0')
                fig_folder = fig_folder / data['date']
                #plot_all_snips(data, fig_folder)
                #plot_all_csd(data, fig_folder)
                fig_folder = Path(r'E:\Mulle\Syt7\Figures\dentate_spikes\ds_detection\local_min')
                plot_all_local_min(data, fig_folder)

                for d in all_data:
                    t0 = np.where(d['ts_snips'] == 0)[0]
                    csd_t0 = np.squeeze(d['csd_snips'][:, :, t0])
                    d['csd_t0'] = csd_t0
                    matching_min, counts = match_local_minima(csd_t0, d['ml_index'])
                    d['matching_min'] = matching_min
                    d['ml_counts'] = counts

                all_sink_ind = [15, 10, 12, 20, 17, None, None, 23, 24]

                d = all_data[8]
                sink_thresh = 0
                sink_ind = 24
                dsm_lfp = np.mean(d['ds_snips'][d['csd_t0'][:, sink_ind] < sink_thresh, :, :], axis=0)
                dsm_csd = np.mean(d['csd_snips'][d['csd_t0'][:, sink_ind] < sink_thresh, :, :], axis=0)
                # dsm_csd = utils.runCSD(dsm_lfp, spacing=50)
                dsl_lfp = np.mean(d['ds_snips'][d['csd_t0'][:, sink_ind] > sink_thresh, :, :], axis=0)
                dsl_csd= np.mean(d['csd_snips'][d['csd_t0'][:, sink_ind] > sink_thresh, :, :], axis=0)
                # dsl_csd = utils.runCSD(dsl_lfp, spacing=50)
                fig, ax = plt.subplots(1, 2)
                levels_ = np.linspace(-100, 100, 50)
                utils.plotCSD(dsm_csd, dsm_lfp, d['ts_snips'], spacing=1,
                              levels_=levels_,
                              xlim=None,
                              cmap='seismic',
                              ax_lbl={'ylabel': 'Layer', 'xlabel': 'Time from dentate spike (ms)'},
                              ax=ax[1])
                utils.plotCSD(dsl_csd, dsl_lfp, d['ts_snips'], spacing=1,
                              levels_=levels_,
                              xlim=None,
                              cmap='seismic',
                              ax_lbl={'ylabel': 'Layer', 'xlabel': 'Time from dentate spike (ms)'},
                              ax=ax[0])



