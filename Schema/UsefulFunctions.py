import numpy as np
import re
from scipy import signal
from scipy.fft import  rfft, rfftfreq, fft, ifft

def ElimMA (bfile,MAlength):

    """
    Function to eliminate MA in a bfile and computer percentages of every MA length after chosing the length of an MA

    inputs:
    bfile : origiinal scored bfile
    MAlength: length in sec of an MA

    outputs:
    bfileMACor: bfile corrected for MA
    MASpansArray : position in the bfile were microarousals start and finish
    PerMAs : percentage of MAs type in the total number of MAs (from "nwn" to "nw..wn" )
    """

    # find MAs and their spans in the b-file

    NumOfScoredWind = int(np.round(MAlength/4))

    MAMiddle = re.finditer('[n2][w1]{1,' + str(NumOfScoredWind) + '}[n2]',bfile)

    MAEnd = re.finditer('[n2][w1]{1,' + str(NumOfScoredWind) + '}' + '$',bfile)

    MABeg= re.finditer('^' + '[w1]{1,' + str(NumOfScoredWind) + '}[n2]',bfile)

    MASpans = []

    for idx, ma in enumerate(MAMiddle):

        Beg = list(ma.span())[0]+1
        End = list(ma.span())[1]-1
        MASpans.append([Beg,End])

    for idx, ma in enumerate(MABeg):

        Beg = list(ma.span())[0]
        End = list(ma.span())[1]-1
        MASpans.append([Beg,End])

    for idx, ma in enumerate(MAEnd):

        Beg = list(ma.span())[0]+1
        End = list(ma.span())[1]
        MASpans.append([Beg,End])

    # compute the percentage of MAs type in the total number of MAs

    PerMAs = []

    # all microarousals

    MAMiddleAll = re.findall('[n2][w1]{1,' + str(NumOfScoredWind) + '}[n2]',bfile)

    MAEndAll = re.findall('[n2][w1]{1,' + str(NumOfScoredWind) + '}' + '$',bfile)

    MABegAll = re.findall('^' + '[w1]{1,' + str(NumOfScoredWind) + '}[n2]',bfile)

    MAAlls = len(MAMiddleAll) + len(MAEndAll) + len(MABegAll)

    # calculate all the type of MAs cand compute the percentage of occurence

    for ma in range(1,NumOfScoredWind+1):

        MAMiddleTemp = re.findall('[n2][w1]{'+ str(ma) + '}[n2]',bfile)

        MAEndTemp = re.findall('[n2][w1]{' + str(ma) + '}' + '$',bfile)

        MABegTemp = re.findall('^' + '[w1]{' + str(ma) + '}[n2]',bfile)

        if not(MAAlls == 0):

            PerTemp = 100*(len(MAMiddleTemp) + len(MAEndTemp) + len(MABegTemp))/MAAlls
        else:
            PerTemp = np.nan

        PerMAs.append(PerTemp)

    # Correct the b-file by removing the MAs by replacing them with an "n" using the previously computed spans

    MASpansArray =  np.array(MASpans)

    bfileMACor = bfile

    for i, ma in enumerate(MASpansArray):

        Temp = bfileMACor[0:ma[0]] + 'n' * (ma[1]-ma[0])  + bfileMACor[ma[1]:]

        bfileMACor = Temp

    return bfileMACor, MASpansArray, PerMAs


def deltaDynamics (bfile,bstart,trace, windows, Fs):

    """
    Function to calculate the delta power dynamics, it will split the session in windows of equal time spent in NREM and calculate
    one value of the delta power band in the middle of each window

    inputs:
    bfile : origiinal scored bfile
    bstart: start of the b-file for calculation
    trace : the trace to calculate the delta power (usually bipolarised EEG)
    windows: how many windows to calculate depending on the recording (for 12h baseline recordings 12 windows for example)

    outputs:
    deltaPowerPointsWakeArray:  delta power dynamics during wake
    deltaPowerPointsNREMArray: delta power dynamics during NREM
    deltaPowerPointsREMArray: delta power dynamics during REM
    timepoints: the timepoints of the calculation
    """


    b = bfile
    startPoints = []
    endPoints = []

    TotalTimeinNREM = 0

    #find the total time in NREM

    NoRemBouts1 = len(re.findall('n',b))
    NoRemBouts2 = len(re.findall('2',b))

    NoRemAll = (NoRemBouts1 + NoRemBouts2)*4/3600

    # divide the total time in NREM in equal windwows

    NoRemPerWindow = NoRemAll/windows

    EpochsPerWindow = int(np.floor(NoRemPerWindow*3600/4))

    #split the b file in windows of equal time of NREM and find the timepoints of splitting

    end = 0
    start = 0

    while start <= len(b):

        EndNotFound = True

        while end <= len(b) and EndNotFound:

            bToCheck = b[start:end]

            NoRemInCheck1 = len(re.findall('n',bToCheck))
            NoRemInCheck2 = len(re.findall('2',bToCheck))

            NoRemInCheck =   NoRemInCheck1 +   NoRemInCheck2

            if NoRemInCheck == EpochsPerWindow:

                startPoints.append(start)
                endPoints.append(end)
                EndNotFound = False

            else:

                end = end + 1

        start = end + 1

    startPoints.append(endPoints[-1] + 1)
    endPoints.append(len(b))

    if len(startPoints) > windows:
        startPoints = startPoints[:-1]
        endPoints = endPoints[:-1]

    startPoints[0] = bstart


    deltaPowerPointsWake = []
    deltaPowerPointsNREM = []
    deltaPowerPointsREM = []

    eeg = trace

    #filter the bipolarised EEG

    bf, af = signal.cheby2(6, 40, 0.5/500, btype ='high', analog=False, output='ba')

    eegfilt = signal.filtfilt(bf, af, eeg)

    timePosition = []

    # transform the scoring windows in time of recordings to align with the EEG
    time = 0
    scoringWindowSec = 4
    v_time = []
    for i,s in enumerate(b):
        v_time.append(time)
        time = time + scoringWindowSec*1000


    # for every window of equal NREM find triplets to calculate the FFT

    for (beg,end) in zip(startPoints,endPoints):

        timePosition.append(np.mean([beg,end])*4/3600)

        bPoints = b[beg:(end+1)]
        v_timePoints = v_time[beg:(end+1)]
        eegPoints =  eegfilt[v_timePoints[0]:(v_timePoints[-1] + Fs*4)]

        NTriplNum = len(re.findall('(?=nnn)',bPoints)) # add + 1
        NTriplNREMbouts = re.finditer('(?=nnn)',bPoints)

        RTriplNum = len(re.findall('(?=rrr)',bPoints))
        RTriplREMbouts = re.finditer('(?=rrr)',bPoints)

        WTriplNum = len(re.findall('(?=www)',bPoints))
        WTriplWakebouts = re.finditer('(?=www)',bPoints)

        NREMIndex = np.zeros(NTriplNum)
        NREMtimes = np.zeros([NTriplNum,2])

        REMIndex = np.zeros(RTriplNum)
        REMtimes = np.zeros([RTriplNum,2])

        WakeIndex = np.zeros(WTriplNum)
        Waketimes = np.zeros([WTriplNum,2])

        EEGNREM = []
        EEGREM = []
        EEGWake = []

        if not (NTriplNum == 0):
            for idx, nrembout in enumerate(NTriplNREMbouts):

                NREMIndex[idx] = int(nrembout.span()[0])+1
                NREMtimes[idx] = [v_time[int(NREMIndex[idx])],v_time[int(NREMIndex[idx])] + Fs*4]
                EEGNREM.append(eegPoints[int(NREMtimes[idx,0]):int(NREMtimes[idx,1])])

        if not (RTriplNum == 0):
            for idx, rembout in enumerate(RTriplREMbouts):

                REMIndex[idx] = int(rembout.span()[0])+1
                REMtimes[idx] = [v_time[int(REMIndex[idx])],v_time[int(REMIndex[idx])] + Fs*4]
                EEGREM.append(eegPoints[int(REMtimes[idx,0]):int(REMtimes[idx,1])])

        if not (WTriplNum == 0):
            for idx, wakebout in enumerate(WTriplWakebouts):

                WakeIndex[idx] = int(wakebout.span()[0])+1
                Waketimes[idx] = [v_time[int(WakeIndex[idx])],v_time[int(WakeIndex[idx])] + Fs*4]
                EEGWake.append(eegPoints[int(Waketimes[idx,0]):int(Waketimes[idx,1])])


        # compute FFT for triplets
        FFTEpochsNREM = []
        FFTEpochsREM = []
        FFTEpochsWake = []

        if not (NTriplNum == 0):

            for epoch in EEGNREM:

                N = len(epoch)

                FreqEpochNREM = rfftfreq(N, 1 / Fs)

                FFTEpochNREM = np.abs(rfft(epoch-np.mean(epoch)))

                FFTEpochNREM = FFTEpochNREM/(N/2)

                FFTEpochNREM  = FFTEpochNREM**2

                FFTEpochsNREM.append(FFTEpochNREM)

            FFTEpochsArrayNREM = np.array(FFTEpochsNREM)


        if not (RTriplNum == 0):

            for epoch in EEGREM:

                N = len(epoch)

                FreqEpochREM = rfftfreq(N, 1 / Fs)

                FFTEpochREM = rfft(epoch-np.mean(epoch))/(N/2)

                FFTEpochREM  = abs(FFTEpochREM )**2

                FFTEpochsREM.append(FFTEpochREM)

            FFTEpochsArrayREM = np.array(FFTEpochsREM)

        if not (WTriplNum == 0):
            for epoch in EEGWake:

                N = len(epoch)

                FreqEpochWake = rfftfreq(N, 1 / Fs)

                FFTEpochWake = rfft(epoch-np.mean(epoch))/(N/2)

                FFTEpochWake  = abs(FFTEpochWake)**2

                FFTEpochsWake.append(FFTEpochWake)


            FFTEpochsArrayWake = np.array(FFTEpochsWake)

        # get the delta power from the FFT

        deltaLimLow = np.argwhere(FreqEpochNREM==1.5)[0][0]
        deltaLimHigh = np.argwhere(FreqEpochNREM==4)[0][0] + 1

        if not (NTriplNum == 0):

            MeanFFTEpochsArrayNREM = np.mean(FFTEpochsArrayNREM,0)
            deltaPowerPointsNREM.append(np.sum(MeanFFTEpochsArrayNREM[deltaLimLow:deltaLimHigh]))

        else:
            deltaPowerPointsNREM.append(0)


        if not (RTriplNum == 0):

            MeanFFTEpochsArrayREM = np.mean(FFTEpochsArrayREM,0)
            deltaPowerPointsREM.append(np.sum(MeanFFTEpochsArrayREM[deltaLimLow:deltaLimHigh]))
        else:
            deltaPowerPointsREM.append(0)

        if not (WTriplNum == 0):

            MeanFFTEpochsArrayWake = np.mean(FFTEpochsArrayWake,0)
            deltaPowerPointsWake.append(np.sum(MeanFFTEpochsArrayWake[deltaLimLow:deltaLimHigh]))

        else:
            deltaPowerPointsWake.append(0)


    deltaPowerPointsWakeArray = np.array(deltaPowerPointsWake)
    deltaPowerPointsNREMArray = np.array(deltaPowerPointsNREM)
    deltaPowerPointsREMArray = np.array(deltaPowerPointsNREM)

    timePositionArray = np.array(timePosition)

    return deltaPowerPointsWakeArray, deltaPowerPointsNREMArray, deltaPowerPointsREMArray, timePositionArray


def maDynamics (bfile,bstart, windows):
    """
    Function to calculate the delta power dynamics, it will split the session in windows of equal time spent in NREM and calculate
    one value of the delta power band in the middle of each window

    inputs:
    bfile : origiinal scored bfile
    bstart: start of the b-file for calculation
    windows: how many windows to calculate depending on the recording (for 12h baseline recordings 12 windows for example)

    outputs:
    MADensityArray:  density of MA's in every window of equal NREM
    timepoints: the timepoints of the calculation
    """

    b = bfile
    startPoints = []
    endPoints = []

    TotalTimeinNREM = 0

    # calculate total time in NREM

    NoRemBouts1 = len(re.findall('n',b))
    NoRemBouts2 = len(re.findall('2',b))

    NoRemAll = (NoRemBouts1 + NoRemBouts2)*4/3600

    NoRemPerWindow = NoRemAll/windows

    EpochsPerWindow = int(np.floor(NoRemPerWindow*3600/4))

    #split the b file in windows of equal time of NREM and find the timepoints of splitting

    end = 0
    start = 0

    while start <= len(b):

        EndNotFound = True

        while end <= len(b) and EndNotFound:

            bToCheck = b[start:end]

            NoRemInCheck1 = len(re.findall('n',bToCheck))
            NoRemInCheck2 = len(re.findall('2',bToCheck))

            NoRemInCheck =   NoRemInCheck1 +   NoRemInCheck2

            if NoRemInCheck == EpochsPerWindow:

                startPoints.append(start)
                endPoints.append(end)
                EndNotFound = False

            else:

                end = end + 1

        start = end + 1

    startPoints.append(endPoints[-1] + 1)
    endPoints.append(len(b))

    if len(startPoints) > windows:
        startPoints = startPoints[:-1]
        endPoints = endPoints[:-1]

    startPoints[0] = bstart

    timePosition = []

    MADensityPoints = []

    # calcualte the number of MA's in every window of equal NREM

    for (beg,end) in zip(startPoints,endPoints):

        timePosition.append(np.mean([beg,end])*4/3600)

        bPoints = b[beg:(end+1)]

        [statesMACor, MASpans, PerMAs] = ElimMA(bPoints,12)

        MAs =  MASpans.shape[0]

        NoRemBouts1 = len(re.findall('n',bPoints))
        NoRemBouts2 = len(re.findall('2',bPoints))

        NoRemMin = (NoRemBouts1 + NoRemBouts2)*4/60

        MADensityPoints.append(MAs/NoRemMin)

    MADensityArray = np.array(MADensityPoints)

    timePositionArray = np.array(timePosition)

    return MADensityArray, timePositionArray


def CorrectMirrorAnimals (HypREMSDPair,HypMirrorPair, REMSDDur):

        """
        Function to correct the hypnogram of a mouse during a REMSD mirror experiment. The wakenings caused by the motor are corrected in the HypMirrorPair using the "rw" information of
        the HypREMSDPair

        inputs:
        HypREMSDPair : the hypnogram of the mouse going through the REMSD while its pair goes through the HypMirrorPair
        HypMirrorPair : the hypnogram of the mouse going through the mirror experiment while its pair is the HypREMSDPair animal
        REMSSDDur: the duration in hours of the REMSD

        outputs:
        HypMirrorPairCor:  the corrected hypnogram of the HypMirrorPair animal
        """

        Hyp1 = HypREMSDPair
        Hyp2 = HypMirrorPair

        REMSDHours = int(REMSDDur*3600/4)

        bfileREMSD = Hyp1[0:REMSDHours]

        RWs = re.finditer('[r3][w1]',bfileREMSD)

        RWsSpans = []

        for idx, rw in enumerate(RWs):

            Beg = list(rw.span())[0]
            End = list(rw.span())[1]-1
            RWsSpans.append([Beg,End])

        HypToPlot1 = np.zeros(len(Hyp1))
        HypToPlot2 = np.zeros(len(Hyp2))
        HypToPlotCor2 = np.zeros(len(Hyp2))

        Hyp2Cor = list(Hyp2)

        for s in  RWsSpans:

            if (not(Hyp2Cor[s[0]] == 'w' or Hyp2Cor[s[0]] == '1') and
                  (Hyp2Cor[np.min([s[0]+1,len(Hyp2Cor)])] == 'w' or Hyp2Cor[np.min([s[0]+1,len(Hyp2Cor)])] == '1')):

                    CharToCorrect = 'u'
                    i = s[0] + 1

                    if (s[0] + 1) < len(Hyp2Cor):

                        while (Hyp2Cor[np.min([i,len(Hyp2Cor)-1])] == 'w' or Hyp2Cor[np.min([i,len(Hyp2Cor)-1])] == '1')  and i< len(Hyp2Cor):
                            Hyp2Cor[i] = CharToCorrect
                            i += 1


            elif (not(Hyp2Cor[np.max([0,s[0]-1])] == 'w' or Hyp2Cor[np.max([0,s[0]-1])] == '1') and
                  (Hyp2Cor[s[0]] == 'w' or Hyp2Cor[s[0]] == '1')):


                        CharToCorrect = 'u'
                        i = s[0]

                        while (Hyp2Cor[np.min([i,len(Hyp2Cor)-1])] == 'w' or Hyp2Cor[np.min([i,len(Hyp2Cor)-1])] == '1') and i< len(Hyp2Cor):
                            Hyp2Cor[i] = CharToCorrect
                            i += 1

                        Found = True


            elif (not(Hyp2Cor[np.max([s[0]-2,0])] == 'w' or Hyp2Cor[np.max([s[0]-2,0])] == '1') and
                  (Hyp2Cor[np.max([s[0]-1,0])] == 'w' or Hyp2Cor[np.max([s[0]-1,0])] == '1') and
                  (Hyp2Cor[np.max([s[0],0])] == 'w' or Hyp2Cor[np.max([s[0],0])] == '1')):


                        CharToCorrect = 'u'

                        i = s[0] - 1

                        if ((s[0]-1) >= 0):

                            while (Hyp2Cor[np.min([i,len(Hyp2Cor)-1])] == 'w' or Hyp2Cor[np.min([i,len(Hyp2Cor)-1])] == '1') and i< len(Hyp2Cor):
                                Hyp2Cor[i] = CharToCorrect
                                i += 1



            elif (not(Hyp2Cor[np.min([s[0]+2,len(Hyp2Cor)-1])] == 'w' or Hyp2Cor[np.min([s[0]+2,len(Hyp2Cor)-1])] == '1') and
                  (Hyp2Cor[np.min([s[0]+3,len(Hyp2Cor)-1])] == 'w' or Hyp2Cor[np.min([s[0]+3,len(Hyp2Cor)-1])] == '1')):



                        CharToCorrect = 'u'

                        Hyp2Cor[s[0]:np.min([s[0]+3,len(Hyp2Cor)])] = list('w'*3)

                        i = s[0]

                        while (Hyp2Cor[np.min([i,len(Hyp2Cor)-1])] == 'w' or Hyp2Cor[np.min([i,len(Hyp2Cor)-1])] == '1')  and i< len(Hyp2Cor):
                            Hyp2Cor[i] = CharToCorrect

                            i += 1

            elif (not(Hyp2Cor[np.min([s[0]+1,len(Hyp2Cor)-1])] == 'w' or Hyp2Cor[np.min([s[0]+1,len(Hyp2Cor)-1])] == '1') and
                  (Hyp2Cor[np.min([s[0]+2,len(Hyp2Cor)-1])] == 'w' or Hyp2Cor[np.min([s[0]+2,len(Hyp2Cor)-1])] == '1')):



                        CharToCorrect = 'u'


                        Hyp2Cor[s[0]:np.min([s[0]+2,len(Hyp2Cor)])] = list('w'*2)


                        i = s[0]

                        while (Hyp2Cor[np.min([i,len(Hyp2Cor)-1])] == 'w' or Hyp2Cor[np.min([i,len(Hyp2Cor)-1])] == '1') and i < len(Hyp2Cor):
                                Hyp2Cor[i] = CharToCorrect
                                i += 1




        Hyp2CorStr = ""

        Hyp2CorFinal = Hyp2CorStr.join(Hyp2Cor)

        HypMirrorPairCor = Hyp2CorFinal

        return HypMirrorPairCor


def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


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


def binning(s,nbins = 10):
    """
    Function which provides mean values of a signal in a defined nbins

    inputs:
    s : the signal to be binned
    nbins: number of bins to compute
    outputs:
    sBinned: binned signal

    """

    lengthPerBin = int(np.ceil(s.shape[0] / nbins))
    sBinned = []

    for b in range(nbins):

        if ((b + 1) * lengthPerBin <= s.shape[0]):

            currentBin = s[(b * lengthPerBin):((b + 1) * lengthPerBin)]
            sBinned.append(np.mean(currentBin))

        else:
            currentBin = s[(b * lengthPerBin):]
            sBinned.append(np.mean(currentBin))

    return  np.array(sBinned)

def MGT(v_Signal,fs,f1,f2,step):

    cycles = 4
    v_freq = np.arange(f1, f2 + step, step)
    s_Len = v_Signal.shape[0]
    s_HalfLen = int(np.floor(s_Len / 2) + 1)

    v_YFFT = fft(v_Signal, v_Signal.shape[0])

    v_WAxis = (2 * np.pi / s_Len) * np.arange(0, s_Len)

    v_WAxis = v_WAxis * fs
    v_WAxisHalf = v_WAxis[0:(s_HalfLen)]
    m_Transform = []
    for i in np.arange(0, v_freq.shape[0]):

        s_ActFrq = v_freq[i]

        dtseg = cycles * (1 / s_ActFrq)

        v_WinFFT = np.zeros([s_Len])

        v_WinFFT[0:s_HalfLen] = np.exp(-0.5 * np.power(v_WAxisHalf - 2 * np.pi * s_ActFrq, 2) * np.power(dtseg, 2))

        v_WinFFT = v_WinFFT * np.sqrt(s_Len) / np.linalg.norm(v_WinFFT, 2)

        m_Transform.append(ifft(v_YFFT * v_WinFFT) / np.sqrt(dtseg))

    m_Transform = np.array(m_Transform)

    return m_Transform

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interpNan(y):

    ycopy = y.copy()

    nans, x = nan_helper(ycopy)

    ycopy[nans] = np.interp(x(nans), x(~nans), ycopy[~nans])

    return ycopy


def BPM(emg,b, Fs, filtertrace =False):

    """
    Function to extract BPM translated from the MATLAB version written by Romain Cardis in 2020

    inputs:
    emg: bipolarised emg signal
    b: scored b file
    Fs: sampling frequency of the emg signal
    filterTrace: flag to decide if the signal will be filtered or not

    output
    bpm: heart rate signal
    """


    if Fs == 1000:
      emg = signal.resample(emg, int(emg.shape[0] / 5))
      Fs = 200

    bf, af = signal.cheby2(6, 40, 25 / (Fs/2), btype='high', analog=False, output='ba')

    emg = signal.filtfilt(bf, af, emg)

    hr = np.abs(np.concatenate(([0],np.diff(emg))))

    timestamps = np.arange(0,emg.shape[0]/Fs,1/Fs)

    TimesOfb = np.arange(0,(len(b))*4,4)

    NREMbouts = re.finditer('(?=nnn)',b)

    PointsNorm = []

    for nrembout in NREMbouts:

     PointStart = int(nrembout.span()[0])
     PointEnd = int(nrembout.span()[0]) + 1

     PointStartSec = TimesOfb[PointStart]
     PointsEndSec = TimesOfb[PointEnd]

     PosStart = findPosFromTimestamps(PointStartSec, timestamps)
     PosEnd = findPosFromTimestamps(PointsEndSec, timestamps)

     PointsNorm.append(hr[PosStart:PosEnd])

    PointsNorm = np.array(PointsNorm)

    PointsNorm =  PointsNorm.flatten()

    hr = (hr - np.mean(PointsNorm)) / np.std(PointsNorm)

    peaks, properties = signal.find_peaks(hr, height = 0.3, distance = 0.08*200)

    peaksToKeep = []

    for i, peak in enumerate(peaks):

        if not(properties['peak_heights'][i] > 10):

            peaksToKeep.append(peak)

    peaksToKeep = np.array(peaksToKeep) / Fs
    bpm = 1. / np.diff(peaksToKeep) * 60

    toLow = bpm<300
    peaksToKeep = peaksToKeep[1::]

    peaksToKeep = np.delete(peaksToKeep, toLow)
    bpm = np.delete(bpm,toLow)

    dbpm = np.concatenate(([0], np.abs(np.diff(bpm))))

    jum = dbpm > 200
    bpm[jum] = np.nan

    bpmFinal = interpNan(bpm)

    x10 = np.linspace(0, int(len(emg) / Fs), int(len(emg) / (Fs/10)))

    bpm10 = np.interp(x10, peaksToKeep, bpmFinal)

    bpm10 = interpNan(bpm10)

    if filtertrace:

        b = signal.firwin(100, 0.025 / 10, pass_zero='lowpass')

        bpm10 = signal.filtfilt(b,1,bpm10)

    return bpm10
