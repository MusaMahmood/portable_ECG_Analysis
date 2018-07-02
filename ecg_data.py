# Adapted from: https://github.com/niekverw/Deep-Learning-Based-ECG-Annotator
# Edited by Musa Mahmood:

import os
import wfdb
import math
import numpy as np
import scipy.stats as st

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('PDF')


# TODO: Create/use rescale-minmax function instead

# Functions
def get_ecg_data(datfile):
    ## convert .dat/q1c to numpy arrays
    recordname = os.path.basename(datfile).split(".dat")[0]
    recordpath = os.path.dirname(datfile)
    cwd = os.getcwd()
    os.chdir(recordpath)  ## somehow it only works if you chdir.

    annotator = 'q1c'
    annotation = wfdb.rdann(recordname, extension=annotator, sampfrom=0, sampto=None, pbdir=None)
    Lstannot = list(zip(annotation.sample, annotation.symbol, annotation.aux_note))

    FirstLstannot = min(i[0] for i in Lstannot)
    LastLstannot = max(i[0] for i in Lstannot) - 1
    print("first-last annotation:", FirstLstannot, LastLstannot)

    record = wfdb.rdsamp(recordname, sampfrom=FirstLstannot, sampto=LastLstannot)  # wfdb.showanncodes()
    annotation = wfdb.rdann(recordname, annotator, sampfrom=FirstLstannot,
                            sampto=LastLstannot)  ## get annotation between first and last.
    annotation2 = wfdb.Annotation(recordname='sel32', extension='niek', sample=(annotation.sample - FirstLstannot),
                                  symbol=annotation.symbol, aux_note=annotation.aux_note)

    Vctrecord = np.transpose(record.p_signals)
    VctAnnotationHot = np.zeros((6, len(Vctrecord[1])), dtype=np.int)
    VctAnnotationHot[5] = 1  ## inverse of the others
    # print("ecg, 2 lead of shape" , Vctrecord.shape)
    # print("VctAnnotationHot of shape" , VctAnnotationHot.shape)
    # print('plotting extracted signal with annotation')
    # wfdb.plotrec(record, annotation=annotation2, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds')

    VctAnnotations = list(zip(annotation2.sample, annotation2.symbol))  ## zip coordinates + annotations (N),(t) etc)
    # print(VctAnnotations)
    for i in range(len(VctAnnotations)):
        # print(VctAnnotations[i]) # Print to display annotations of an ecg
        try:

            if VctAnnotations[i][1] == "p":
                if VctAnnotations[i - 1][1] == "(":
                    pstart = VctAnnotations[i - 1][0]
                if VctAnnotations[i + 1][1] == ")":
                    pend = VctAnnotations[i + 1][0]
                if VctAnnotations[i + 3][1] == "N":
                    rpos = VctAnnotations[i + 3][0]
                    if VctAnnotations[i + 2][1] == "(":
                        qpos = VctAnnotations[i + 2][0]
                    if VctAnnotations[i + 4][1] == ")":
                        spos = VctAnnotations[i + 4][0]
                    for ii in range(0, 8):  ## search for t (sometimes the "(" for the t  is missing  )
                        if VctAnnotations[i + ii][1] == "t":
                            tpos = VctAnnotations[i + ii][0]
                            if VctAnnotations[i + ii + 1][1] == ")":
                                tendpos = VctAnnotations[i + ii + 1][0]
                                # 				#print(ppos,qpos,rpos,spos,tendpos)
                                VctAnnotationHot[0][pstart:pend] = 1  # P segment
                                VctAnnotationHot[1][
                                pend:qpos] = 1  # part "nothing" between P and Q, previously left unnanotated, but categorical probably can't deal with that
                                VctAnnotationHot[2][qpos:rpos] = 1  # QR
                                VctAnnotationHot[3][rpos:spos] = 1  # RS
                                VctAnnotationHot[4][spos:tendpos] = 1  # ST (from end of S to end of T)
                                VctAnnotationHot[5][
                                pstart:tendpos] = 0  # tendpos:pstart becomes 1, because it is inverted above
        except IndexError:
            pass

    Vctrecord = np.transpose(Vctrecord)  # transpose to (timesteps,feat)
    VctAnnotationHot = np.transpose(VctAnnotationHot)
    os.chdir(cwd)
    return Vctrecord, VctAnnotationHot


def splitseq(x, n, o):
    # split seq; should be optimized so that remove_seq_gaps is not needed.
    upper = math.ceil(x.shape[0] / n) * n
    print("splitting on", n, "with overlap of ", o, "total datapoints:", x.shape[0], "; upper:", upper)
    for i in range(0, upper, n):
        # print(i)
        if i == 0:
            padded = np.zeros((o + n + o, x.shape[1]))  ## pad with 0's on init
            padded[o:, :x.shape[1]] = x[i:i + n + o, :]
            xpart = padded
        else:
            xpart = x[i - o:i + n + o, :]
        if xpart.shape[0] < i:
            padded = np.zeros((o + n + o, xpart.shape[1]))  ## pad with 0's on end of seq
            padded[:xpart.shape[0], :xpart.shape[1]] = xpart
            xpart = padded

        xpart = np.expand_dims(xpart, 0)  ## add one dimension; so that you get shape (samples,timesteps,features)
        try:
            xx = np.vstack((xx, xpart))
        except UnboundLocalError:  ## on init
            xx = xpart
    print("output: ", xx.shape)
    return (xx)


def remove_seq_gaps(x, y):
    # remove parts that are not annotated <- not ideal, but quickest for now.
    window = 150
    c = 0
    cutout = []
    include = []
    print("filterering.")
    print("before shape x,y", x.shape, y.shape)
    for i in range(y.shape[0]):

        c = c + 1
        if c < window:
            include.append(i)
        if sum(y[i, 0:5]) > 0:
            c = 0
        if c >= window:
            # print ('filtering')
            pass
    x, y = x[include, :], y[include, :]
    print(" after shape x,y", x.shape, y.shape)
    return (x, y)


def normalizesignal(x):
    x = st.zscore(x, ddof=0)
    return x


def normalizesignal_array(x):
    for i in range(x.shape[0]):
        x[i] = st.zscore(x[i], axis=0, ddof=0)
    return x

def plotecg(x, y, begin, end):
    # helper to plot ecg
    plt.figure(1, figsize=(11.69, 8.27))
    plt.subplot(211)
    plt.plot(x[begin:end, 0])
    plt.subplot(211)
    plt.plot(y[begin:end, 0])
    plt.subplot(211)
    plt.plot(y[begin:end, 1])
    plt.subplot(211)
    plt.plot(y[begin:end, 2])
    plt.subplot(211)
    plt.plot(y[begin:end, 3])
    plt.subplot(211)
    plt.plot(y[begin:end, 4])
    plt.subplot(211)
    plt.plot(y[begin:end, 5])

    plt.subplot(212)
    plt.plot(x[begin:end, 1])
    plt.show()


def plotecg_validation(x, y_true, y_pred, begin, end):
    # helper to plot ecg
    plt.figure(1, figsize=(11.69, 8.27))
    plt.subplot(211)
    plt.plot(x[begin:end, 0])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 0])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 1])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 2])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 3])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 4])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 5])

    plt.subplot(212)
    plt.plot(x[begin:end, 1])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 0])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 1])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 2])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 3])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 4])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 5])
