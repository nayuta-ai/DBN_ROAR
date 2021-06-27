import numpy as np
import mne
import scipy.stats
import matplotlib.pyplot as plt

def topomap(data):
    norm = scipy.stats.zscore(data)
    biosemi_montage = mne.channels.make_standard_montage('biosemi32')
    n_channels = len(biosemi_montage.ch_names)
    fake_info = mne.create_info(ch_names=biosemi_montage.ch_names,sfreq=128.,
                                ch_types='eeg')
    rng = np.random.RandomState(0)
    data_plot = norm[0:32,0:1]
    fake_evoked = mne.EvokedArray(data_plot, fake_info)
    fake_evoked.set_montage(biosemi_montage)
    mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info,
                     show=False)

def curve(list):
    plt.plot(list)
    plt.xlabel('trial')
    plt.ylabel('accuracy')
    plt.title('ROAR curve')
    plt.grid(True)
    plt.savefig("ROAR_curve.png")
    plt.show()