
import numpy as np
import re
from scipy import signal
from scipy import interpolate
from scipy import optimize

def elim_artefacts(b):

    """
    Function to eliminate artifacts from a b-file usually scored as 1,2,3
    inputs
    b: the b-file as scored
    """
    b_noArtefacts = b.replace('1', 'w')
    b_noArtefacts = b_noArtefacts.replace('2', 'n')
    b_noArtefacts = b_noArtefacts.replace('3', 'r')

    return b_noArtefacts


def elim_MA(b):

    """
    Function to eliminate artifacts MA from a b-file
    inputs
    b: the b-file as scored
    """

    ma_1 = re.finditer('(?=nwn)', b)
    ma_2 = re.finditer('(?=nwwn)', b)
    ma_3 = re.finditer('(?=nwwwn)', b)

    array_ma1 = []
    for ma in ma_1:
        array_ma1.append(ma.start())
    array_ma1 = np.array(array_ma1)

    array_ma2 = []
    for ma in ma_2:
        array_ma2.append(ma.start())
    array_ma2 = np.array(array_ma2)

    array_ma3 = []
    for ma in ma_3:
        array_ma3.append(ma.start())
    array_ma3 = np.array(array_ma3)

    ma_index = np.concatenate((array_ma1, array_ma2, array_ma3))
    ma_index = sorted(ma_index)

    b_noMA = b

    b_noMA = b_noMA.replace('nwn', 'nnn')
    b_noMA = b_noMA.replace('nwwn', 'nnnn')
    b_noMA = b_noMA.replace('nwwwn', 'nnnnn')

    return ma_index, b_noMA


def b_num(b, time):

    """
    Function to create a b file with numbers 1,2,3 replacing r,n and w
    at the same time the final b-file is matched in sampling frequency to the
    acquired signal of fiber photometry as denoted by the timestamps (time)
    inputs
    b: the b-file as scored
    time: the timestamps of acquisition for the fiber photometry signal
    """

    b_numerized = np.zeros(len(time), dtype=int)

    timeInb = np.arange(0,(len(b)-1)*4,4)

    for i,t in enumerate(time):

        Diff = np.abs(timeInb - t)
        Pos = np.where(Diff == np.min(Diff))[0][0]

        if b[Pos] == 'w':
            b_numerized[i] = 3

        elif b[Pos] == 'n':
            b_numerized[i] = 2

        elif b[Pos] == 'r':
            b_numerized[i] = 1

    return b_numerized

def Get_envelope_distance(x, y):
    """
    Calculates the envelope of the interpolated data using a distance criterion for the findpeaks steps

    inputs:
    x: the signal for which the envelope needs to be computed

    outputs:
    y: the distance criterion in number of points
    """

    peaks, _ = signal.find_peaks(x.squeeze(), height=0,
                                 distance=y)  # find the peaks (above 0) in the interpolated data
    troughs, _ = signal.find_peaks(-x.squeeze(), distance=y)  # find the troughts in the interpolated data

    # Fit suitable models to the data. Here I am using cubic splines.
    u_p = interpolate.CubicSpline(peaks, x[peaks])
    l_p = interpolate.CubicSpline(troughs, x[troughs])
    Upper_envelope = [u_p(i) for i in range(x.shape[0])]
    Lower_envelope = [l_p(i) for i in range(x.shape[0])]

    return peaks, troughs, Upper_envelope, Lower_envelope

def Get_envelope(x):
    """
    Calculates the envelop of the interpolated data.

    inputs:
    x: the signal for which the envelope needs to be computed
    """

    peaks, _ = signal.find_peaks(x.squeeze(), height=0)  # find the peaks (above 0) in the interpolated data
    troughs, _ = signal.find_peaks(-x.squeeze())  # find the troughts in the interpolated data

    # Fit suitable models to the data. Here I am using cubic splines.
    u_p = interpolate.CubicSpline(peaks, x[peaks])
    l_p = interpolate.CubicSpline(troughs, x[troughs])
    Upper_envelope = [u_p(i) for i in range(x.shape[0])]
    Lower_envelope = [l_p(i) for i in range(x.shape[0])]

    return peaks, troughs, Upper_envelope, Lower_envelope

def Expfunc(x, a, b):
    return a * np.exp(b * x)

def findPosFromTimestamps(timeStamps,timeToLoc):
    """
    Function to gives you a position in a vector where a specific timepoint is located in recorded timestamps

    inputs:
    timeStamps : the vector of recorded timestamps (in sec)
    timeToLoc: time and to locate in timestamps (in sec)
    outputs:
    pos:  position in recorded timestamps
    """

    DiffTime = np.abs(timeStamps - timeToLoc)
    pos = np.where(DiffTime == np.min(DiffTime))[0][0]

    return pos

def func(x, a, b):
    """
    Exponential function to fit using the polyfit function
    """
    return a * np.exp(b * x)


def DynamicRange(signal):
    """
    Calculation of the dynamic range of the signal using a min/max substraction
    """
    return np.max(signal)-np.min(signal)


def PeakDetection(dff,b_file, time):
    """
    Extraction of LC peak activity from fiber photometry signals based on the algorithm described
    in Osorio-Forero, Foustoukos et al. 2024.Inspired by the MATLAB version written by Dr. Alejandro Osorio-Forero
    and adapted to Python and the Doric system acquisition by Dr.Georgios Foustoukos and Paola Milanese in May 2024

    inputs:
    dff: the DFF signal of LC during
    b_file: the b_file indicating the sleep state with the letter w:Wake, n:NREMS, r:REM scored at 4s windows
    time: timestamps of the dff acquisition by the Doric system

    outputs:
    peaks_dff2_index: positions of LC peaks
    PeaksFlag: boolean matrix with 0 if a peak was non associated with a MA and 1 if it did.
     """

    sampling_freq = int(np.round(1 / np.mean(np.diff(time))))
    wakelengthbeforeNREM = 60

    # eliminate artifacts, MA, and get a b file in a numerised version and matched to the signal length
    b_noArtefacts = elim_artefacts(b_file)
    ma_index, b_noMA_noArt = elim_MA(b_noArtefacts)
    b_numerized = b_num(b_noMA_noArt, time)

    # artificially setting the last samples of the trace to wake to avoid artefacts in the dff.
    b_numerized[-(1*sampling_freq):] = np.ones((1*sampling_freq), dtype='int') * 3
    b_numerized[0:(1*sampling_freq)] = np.ones((1*sampling_freq), dtype='int') * 3
    dff[-(2*sampling_freq):] = np.ones(2*sampling_freq, dtype='int') * 3
    dff[0:2*sampling_freq] = np.ones(2*sampling_freq, dtype='int') * 3

    # filtering dff with a low-pass of 0.1 cutoff freq and a second one of 0.5
    high_lowpass = signal.firwin(1000, 0.5 / (sampling_freq / 2), pass_zero='lowpass')
    low_lowpass = signal.firwin(1000, 0.1 / (sampling_freq / 2), pass_zero='lowpass')

    dff_high = signal.filtfilt(high_lowpass, 1, dff)
    dff_low = signal.filtfilt(low_lowpass, 1, dff)

    # compute the envelope of the signal (low and upper)
    peaks, troughs, dff_upper_enveloppe, dff_lower_enveloppe = Get_envelope(dff_low)

    # flatten the signal using its lower envelope
    signal_dff = (dff_high - dff_lower_enveloppe)

    # find all peaks in the signal
    peaks_dff1_index, _ = signal.find_peaks(signal_dff)
    peaks1 = dff[peaks_dff1_index]
    prominences1 = signal.peak_prominences(signal_dff, peaks_dff1_index)[0]

    # get rid of the peaks using a prominence criterion
    peaks_dff2_index, _ = signal.find_peaks(signal_dff, prominence=np.percentile(prominences1, 60))
    peaks2 = dff[peaks_dff2_index]
    prominences2 = signal.peak_prominences(signal_dff, peaks_dff2_index)[0]

    # get rid of the peaks using two more criteria, one of prominence and one of amplitude
    todelete = np.zeros(len(peaks2))
    for idx_peak in range(len(peaks2)):
        if (prominences2[idx_peak] < 0.25 * np.nanmax(signal_dff[peaks_dff2_index])) & (
                signal_dff[peaks_dff2_index[idx_peak]] < np.nanmax(signal_dff) * 0.2):
            todelete[idx_peak] = 1

    peaks3 = np.delete(peaks2, todelete == 1)
    prominences3 = np.delete(prominences2, todelete == 1)
    peaks_dff2_index = np.delete(peaks_dff2_index, todelete == 1)

    # detect NREM (90sec) bouts after long wake (60sec)
    longNREM_re = re.finditer('w{15}n{24,}', b_noMA_noArt)
    longNREM_start = []
    longNREM_end = []

    for long_nonrem in longNREM_re:
        span = long_nonrem.span()
        longNREM_start.append(span[0])
        longNREM_end.append(span[1])

    longNREM_start = np.array(longNREM_start)
    longNREM_end = np.array(longNREM_end)

    # get rid of bouts at the edges
    if longNREM_end[-1] > len(b_noMA_noArt) * .99:
        longNREM_start = np.delete(longNREM_start, -1)
        longNREM_end = np.delete(longNREM_end, -1)

    # use the highly filtered dff signal
    dff_low_corrected = dff_low - np.min(dff_low)

    # extract the wake and NREM bouts from the dff signal and their timestamps
    waketoNREMbouts_dff = []
    timesWaketoNREbouts = []
    currentbout = []
    for idx_Bout in range(len(longNREM_start)):
        timeStart = findPosFromTimestamps(time, longNREM_start[idx_Bout] * 4)
        timeStop = findPosFromTimestamps(time, longNREM_end[idx_Bout] * 4)

        currentbout = dff_low_corrected[timeStart:timeStop]

        currentbout = currentbout - np.min(currentbout)
        waketoNREMbouts_dff.append(currentbout)

        timesWaketoNREbouts.append(time[timeStart:timeStop])

    # compute the lower envelope of the wake-NREM bout and fit an exponential curve in order to compute tau
    tau = np.zeros(len(longNREM_start))
    for bout in range(len(longNREM_start)):
        # print(bout)

        time_vec = timesWaketoNREbouts[bout] - timesWaketoNREbouts[bout][0]

        # the envelope is computed using a find-peaks function with a distance criterion (96 sec of distance between two peaks)
        peaks, troughs, Upper_envelope, Lower_envelope = Get_envelope_distance(waketoNREMbouts_dff[bout],
                                                                               96 * sampling_freq)

        Lower_envelope = np.array(Lower_envelope)

        timeNREMAfterLongWake = time_vec[time_vec > wakelengthbeforeNREM]
        EnvDffNREMAfterLongWake = Lower_envelope[time_vec > wakelengthbeforeNREM]

        a, b = optimize.curve_fit(func, timeNREMAfterLongWake, EnvDffNREMAfterLongWake, p0=(1, 1e-6))

        # plt.plot( time_vec[time_vec>wakelengthbeforeNREM],func(time_vec[time_vec>wakelengthbeforeNREM], a[0], a[1]))

        tau[bout] = -1 / (a[1])
        # print(tau[bout])

    # delete taus with outlier values and get the mean value
    todelete = np.zeros(len(tau))
    for idx_bout in range(len(tau)):
        if tau[idx_bout] < 0 or tau[idx_bout] > 1000:
            todelete[idx_bout] = 1
    tau = np.delete(tau, todelete == 1)
    av_tau = np.mean(tau)

    # delete peaks which are less than 1.5*mean(tau) close to a preceding wake event
    time_vec2 = time - time[0]

    count = 0
    todelete = np.zeros(len(peaks_dff2_index))
    for idxpeak in peaks_dff2_index:
        current = np.zeros(dff.shape[0])
        current[np.logical_and(time_vec2 < time_vec2[idxpeak], time_vec2 > (time_vec2[idxpeak] - av_tau * 1.5))] = 1
        if any(b_numerized[current == 1] > 2):
            todelete[count] = 1
        count += 1
    peaks3 = np.delete(peaks3, todelete == 1)
    prominences3 = np.delete(prominences3, todelete == 1)
    peaks_dff2_index = np.delete(peaks_dff2_index, todelete == 1)

    # classify the peaks in two classes depending on the concomitant occurrence of a MA (within 5s of the peak point)
    timeInb = np.arange(0, (len(b_file) - 1) * 4, 4)
    MATimes = timeInb[ma_index]
    PeakTimes = time[peaks_dff2_index]
    PeaksFlag = []

    for p in PeakTimes:

        TimeDiff = np.abs(MATimes - p)

        if np.sum(TimeDiff < 5) > 0:

            PeaksFlag.append(1)
        else:
            PeaksFlag.append(0)

    return peaks_dff2_index, PeaksFlag
