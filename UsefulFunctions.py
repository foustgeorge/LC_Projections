# import the packages need
import re
import plotly.graph_objects as go
from scipy.fft import rfft, rfftfreq, fft, ifft
from scipy import interpolate
from scipy import signal

def b_num(b, time):
    """
    Convert sleep state string to numerical format and match sampling frequency
    to fiber photometry timestamps.

    This function creates a numerical b-file where sleep states are converted:
    'w' (wake) -> 3, 'n' (NREM) -> 2, 'r' (REM) -> 1

    inputs:
    b: the b-file as scored (string with sleep state characters)
    time: the timestamps of acquisition for the fiber photometry signal

    outputs:
    b_numerized: numerical array matching fiber photometry sampling rate
    """

    # Initialize numerical b-file array matching photometry timestamps
    b_numerized = np.zeros(len(time), dtype=int)

    # Create time array for b-file (each character represents 4 seconds)
    timeInb = np.arange(0, (len(b)-1)*4, 4)

    # For each photometry timestamp, find closest b-file time point
    for i, t in enumerate(time):
        # Find minimum time difference
        Diff = np.abs(timeInb - t)
        Pos = np.where(Diff == np.min(Diff))[0][0]

        # Convert sleep state character to number
        if b[Pos] == 'w':
            b_numerized[i] = 3      # Wake = 3
        elif b[Pos] == 'n':
            b_numerized[i] = 2      # NREM = 2
        elif b[Pos] == 'r':
            b_numerized[i] = 1      # REM = 1

    return b_numerized


def ElimMA(bfile, MAlength):
    """
    Eliminate microarousals (MA) from a b-file and compute MA statistics.

    Microarousals are brief awakenings during sleep, defined as wake periods
    of specified duration surrounded by NREM/stage 2 sleep.

    inputs:
    bfile: original scored b-file string
    MAlength: maximum length of microarousal in seconds

    outputs:
    bfileMACor: b-file with microarousals replaced by 'n' (NREM)
    MASpansArray: array of [start, end] positions where MAs occur
    PerMAs: percentage breakdown of MA types by duration
    """

    # Convert MA length from seconds to number of 4-second epochs
    NumOfScoredWind = int(np.round(MAlength/4))

    # STEP 1: FIND MICROAROUSALS USING REGULAR EXPRESSIONS

    # Find MAs in middle of recording: NREM/stage2 -> wake -> NREM/stage2
    MAMiddle = re.finditer('[n2][w1]{1,' + str(NumOfScoredWind) + '}[n2]', bfile)

    # Find MAs at end of recording: NREM/stage2 -> wake -> end
    MAEnd = re.finditer('[n2][w1]{1,' + str(NumOfScoredWind) + '}' + '$', bfile)

    # Find MAs at beginning of recording: start -> wake -> NREM/stage2
    MABeg = re.finditer('^' + '[w1]{1,' + str(NumOfScoredWind) + '}[n2]', bfile)

    # Extract start and end positions of all microarousals
    MASpans = []

    # Process middle MAs (exclude surrounding NREM epochs)
    for idx, ma in enumerate(MAMiddle):
        Beg = list(ma.span())[0] + 1  # Start after first NREM
        End = list(ma.span())[1] - 1  # End before last NREM
        MASpans.append([Beg, End])

    # Process beginning MAs
    for idx, ma in enumerate(MABeg):
        Beg = list(ma.span())[0]      # Include from start
        End = list(ma.span())[1] - 1  # End before NREM
        MASpans.append([Beg, End])

    # Process end MAs
    for idx, ma in enumerate(MAEnd):
        Beg = list(ma.span())[0] + 1  # Start after NREM
        End = list(ma.span())[1]      # Include to end
        MASpans.append([Beg, End])

    # STEP 2: COMPUTE PERCENTAGE BREAKDOWN OF MA TYPES

    PerMAs = []

    # Count all microarousals for percentage calculation
    MAMiddleAll = re.findall('[n2][w1]{1,' + str(NumOfScoredWind) + '}[n2]', bfile)
    MAEndAll = re.findall('[n2][w1]{1,' + str(NumOfScoredWind) + '}' + '$', bfile)
    MABegAll = re.findall('^' + '[w1]{1,' + str(NumOfScoredWind) + '}[n2]', bfile)

    MAAlls = len(MAMiddleAll) + len(MAEndAll) + len(MABegAll)

    # Calculate percentage for each MA duration (1 to NumOfScoredWind epochs)
    for ma in range(1, NumOfScoredWind + 1):
        # Count MAs of exactly 'ma' epochs duration
        MAMiddleTemp = re.findall('[n2][w1]{' + str(ma) + '}[n2]', bfile)
        MAEndTemp = re.findall('[n2][w1]{' + str(ma) + '}' + '$', bfile)
        MABegTemp = re.findall('^' + '[w1]{' + str(ma) + '}[n2]', bfile)

        if not(MAAlls == 0):
            # Calculate percentage of this MA type
            PerTemp = 100 * (len(MAMiddleTemp) + len(MAEndTemp) + len(MABegTemp)) / MAAlls
        else:
            PerTemp = np.nan

        PerMAs.append(PerTemp)

    # STEP 3: CORRECT B-FILE BY REPLACING MAs WITH 'n' (NREM)

    MASpansArray = np.array(MASpans)
    bfileMACor = bfile

    # Replace each MA span with 'n' characters
    for i, ma in enumerate(MASpansArray):
        Temp = bfileMACor[0:ma[0]] + 'n' * (ma[1] - ma[0]) + bfileMACor[ma[1]:]
        bfileMACor = Temp

    return bfileMACor, MASpansArray, PerMAs


def deltaDynamics(bfile, bstart, trace, windows, Fs):
    """
    Calculate delta power dynamics across equal NREM time windows.

    This function splits a recording into windows containing equal amounts of NREM sleep,
    then calculates delta power (1.5-4 Hz) for Wake, NREM, and REM states within each window.

    inputs:
    bfile: original scored b-file string
    bstart: start index of the b-file for calculation
    trace: EEG trace for delta power calculation (usually bipolarized EEG)
    windows: number of windows to create (e.g., 12 for 12-hour recording)
    Fs: sampling frequency of the EEG signal

    outputs:
    deltaPowerPointsWakeArray: delta power dynamics during wake
    deltaPowerPointsNREMArray: delta power dynamics during NREM
    deltaPowerPointsREMArray: delta power dynamics during REM
    timePositionArray: timepoints of calculations (hours)
    """

    b = bfile
    startPoints = []
    endPoints = []

    # STEP 1: CALCULATE TOTAL NREM TIME AND DIVIDE INTO EQUAL WINDOWS

    # Count total NREM epochs ('n' and '2' states)
    NoRemBouts1 = len(re.findall('n', b))
    NoRemBouts2 = len(re.findall('2', b))

    # Convert to hours (each epoch = 4 seconds)
    NoRemAll = (NoRemBouts1 + NoRemBouts2) * 4 / 3600

    # Calculate NREM time per window
    NoRemPerWindow = NoRemAll / windows
    EpochsPerWindow = int(np.floor(NoRemPerWindow * 3600 / 4))

    # STEP 2: FIND WINDOW BOUNDARIES WITH EQUAL NREM TIME

    end = 0
    start = 0

    # Iterate through b-file to find windows with equal NREM content
    while start <= len(b):
        EndNotFound = True

        while end <= len(b) and EndNotFound:
            bToCheck = b[start:end]

            # Count NREM epochs in current window
            NoRemInCheck1 = len(re.findall('n', bToCheck))
            NoRemInCheck2 = len(re.findall('2', bToCheck))
            NoRemInCheck = NoRemInCheck1 + NoRemInCheck2

            if NoRemInCheck == EpochsPerWindow:
                # Found window with correct NREM amount
                startPoints.append(start)
                endPoints.append(end)
                EndNotFound = False
            else:
                end = end + 1

        start = end + 1

    # Add final window
    startPoints.append(endPoints[-1] + 1)
    endPoints.append(len(b))

    # Limit to requested number of windows
    if len(startPoints) > windows:
        startPoints = startPoints[:-1]
        endPoints = endPoints[:-1]

    startPoints[0] = bstart

    # Initialize output arrays
    deltaPowerPointsWake = []
    deltaPowerPointsNREM = []
    deltaPowerPointsREM = []

    eeg = trace

    # STEP 3: FILTER EEG SIGNAL

    # Apply high-pass filter (0.5 Hz) to remove slow drift
    bf, af = signal.cheby2(6, 40, 0.5/500, btype='high', analog=False, output='ba')
    eegfilt = signal.filtfilt(bf, af, eeg)

    timePosition = []

    # Create time vector for b-file epochs (milliseconds for EEG indexing)
    time = 0
    scoringWindowSec = 4
    v_time = []
    for i, s in enumerate(b):
        v_time.append(time)
        time = time + scoringWindowSec * 1000

    # STEP 4: PROCESS EACH WINDOW

    for (beg, end) in zip(startPoints, endPoints):
        # Calculate time position of window center (hours)
        timePosition.append(np.mean([beg, end]) * 4 / 3600)

        # Extract b-file and EEG data for this window
        bPoints = b[beg:(end+1)]
        v_timePoints = v_time[beg:(end+1)]
        eegPoints = eegfilt[v_timePoints[0]:(v_timePoints[-1] + Fs*4)]

        # STEP 5: FIND TRIPLETS OF CONSECUTIVE SAME-STATE EPOCHS

        # Find NREM triplets (nnn)
        NTriplNum = len(re.findall('(?=nnn)', bPoints))
        NTriplNREMbouts = re.finditer('(?=nnn)', bPoints)

        # Find REM triplets (rrr)
        RTriplNum = len(re.findall('(?=rrr)', bPoints))
        RTriplREMbouts = re.finditer('(?=rrr)', bPoints)

        # Find Wake triplets (www)
        WTriplNum = len(re.findall('(?=www)', bPoints))
        WTriplWakebouts = re.finditer('(?=www)', bPoints)

        # Initialize arrays for each state
        NREMIndex = np.zeros(NTriplNum)
        NREMtimes = np.zeros([NTriplNum, 2])
        REMIndex = np.zeros(RTriplNum)
        REMtimes = np.zeros([RTriplNum, 2])
        WakeIndex = np.zeros(WTriplNum)
        Waketimes = np.zeros([WTriplNum, 2])

        EEGNREM = []
        EEGREM = []
        EEGWake = []

        # STEP 6: EXTRACT EEG SEGMENTS FOR EACH STATE

        # Extract NREM EEG segments
        if not (NTriplNum == 0):
            for idx, nrembout in enumerate(NTriplNREMbouts):
                NREMIndex[idx] = int(nrembout.span()[0]) + 1
                NREMtimes[idx] = [v_time[int(NREMIndex[idx])],
                                 v_time[int(NREMIndex[idx])] + Fs*4]
                EEGNREM.append(eegPoints[int(NREMtimes[idx,0]):int(NREMtimes[idx,1])])

        # Extract REM EEG segments
        if not (RTriplNum == 0):
            for idx, rembout in enumerate(RTriplREMbouts):
                REMIndex[idx] = int(rembout.span()[0]) + 1
                REMtimes[idx] = [v_time[int(REMIndex[idx])],
                                v_time[int(REMIndex[idx])] + Fs*4]
                EEGREM.append(eegPoints[int(REMtimes[idx,0]):int(REMtimes[idx,1])])

        # Extract Wake EEG segments
        if not (WTriplNum == 0):
            for idx, wakebout in enumerate(WTriplWakebouts):
                WakeIndex[idx] = int(wakebout.span()[0]) + 1
                Waketimes[idx] = [v_time[int(WakeIndex[idx])],
                                 v_time[int(WakeIndex[idx])] + Fs*4]
                EEGWake.append(eegPoints[int(Waketimes[idx,0]):int(Waketimes[idx,1])])

        # STEP 7: COMPUTE FFT FOR EACH EEG SEGMENT

        FFTEpochsNREM = []
        FFTEpochsREM = []
        FFTEpochsWake = []

        # Compute FFT for NREM epochs
        if not (NTriplNum == 0):
            for epoch in EEGNREM:
                N = len(epoch)
                FreqEpochNREM = rfftfreq(N, 1 / Fs)
                FFTEpochNREM = np.abs(rfft(epoch - np.mean(epoch)))
                FFTEpochNREM = FFTEpochNREM / (N/2)  # Normalize
                FFTEpochNREM = FFTEpochNREM**2       # Power spectrum
                FFTEpochsNREM.append(FFTEpochNREM)

            FFTEpochsArrayNREM = np.array(FFTEpochsNREM)

        # Compute FFT for REM epochs
        if not (RTriplNum == 0):
            for epoch in EEGREM:
                N = len(epoch)
                FreqEpochREM = rfftfreq(N, 1 / Fs)
                FFTEpochREM = rfft(epoch - np.mean(epoch)) / (N/2)
                FFTEpochREM = abs(FFTEpochREM)**2
                FFTEpochsREM.append(FFTEpochREM)

            FFTEpochsArrayREM = np.array(FFTEpochsREM)

        # Compute FFT for Wake epochs
        if not (WTriplNum == 0):
            for epoch in EEGWake:
                N = len(epoch)
                FreqEpochWake = rfftfreq(N, 1 / Fs)
                FFTEpochWake = rfft(epoch - np.mean(epoch)) / (N/2)
                FFTEpochWake = abs(FFTEpochWake)**2
                FFTEpochsWake.append(FFTEpochWake)

            FFTEpochsArrayWake = np.array(FFTEpochsWake)

        # STEP 8: EXTRACT DELTA POWER (1.5-4 Hz)

        # Define delta frequency band indices
        deltaLimLow = np.argwhere(FreqEpochNREM == 1.5)[0][0]
        deltaLimHigh = np.argwhere(FreqEpochNREM == 4)[0][0] + 1

        # Calculate mean delta power for each state
        if not (NTriplNum == 0):
            MeanFFTEpochsArrayNREM = np.mean(FFTEpochsArrayNREM, 0)
            deltaPowerPointsNREM.append(np.sum(MeanFFTEpochsArrayNREM[deltaLimLow:deltaLimHigh]))
        else:
            deltaPowerPointsNREM.append(0)

        if not (RTriplNum == 0):
            MeanFFTEpochsArrayREM = np.mean(FFTEpochsArrayREM, 0)
            deltaPowerPointsREM.append(np.sum(MeanFFTEpochsArrayREM[deltaLimLow:deltaLimHigh]))
        else:
            deltaPowerPointsREM.append(0)

        if not (WTriplNum == 0):
            MeanFFTEpochsArrayWake = np.mean(FFTEpochsArrayWake, 0)
            deltaPowerPointsWake.append(np.sum(MeanFFTEpochsArrayWake[deltaLimLow:deltaLimHigh]))
        else:
            deltaPowerPointsWake.append(0)

    # Convert to numpy arrays
    deltaPowerPointsWakeArray = np.array(deltaPowerPointsWake)
    deltaPowerPointsNREMArray = np.array(deltaPowerPointsNREM)
    deltaPowerPointsREMArray = np.array(deltaPowerPointsNREM)  # Note: this should probably be deltaPowerPointsREM
    timePositionArray = np.array(timePosition)

    return deltaPowerPointsWakeArray, deltaPowerPointsNREMArray, deltaPowerPointsREMArray, timePositionArray


def maDynamics(bfile, bstart, windows):
    """
    Calculate microarousal density dynamics across equal NREM time windows.

    This function splits a recording into windows containing equal amounts of NREM sleep,
    then calculates the density of microarousals (MAs per minute of NREM) in each window.

    inputs:
    bfile: original scored b-file string
    bstart: start index of the b-file for calculation
    windows: number of windows to create

    outputs:
    MADensityArray: density of microarousals in each window (MAs per minute of NREM)
    timePositionArray: timepoints of calculations (hours)
    """

    b = bfile
    startPoints = []
    endPoints = []

    # STEP 1: CALCULATE TOTAL NREM TIME AND DIVIDE INTO EQUAL WINDOWS
    # (Same logic as deltaDynamics function)

    # Count total NREM epochs
    NoRemBouts1 = len(re.findall('n', b))
    NoRemBouts2 = len(re.findall('2', b))
    NoRemAll = (NoRemBouts1 + NoRemBouts2) * 4 / 3600  # Convert to hours

    # Calculate NREM time per window
    NoRemPerWindow = NoRemAll / windows
    EpochsPerWindow = int(np.floor(NoRemPerWindow * 3600 / 4))

    # Find window boundaries with equal NREM content
    end = 0
    start = 0

    while start <= len(b):
        EndNotFound = True

        while end <= len(b) and EndNotFound:
            bToCheck = b[start:end]

            NoRemInCheck1 = len(re.findall('n', bToCheck))
            NoRemInCheck2 = len(re.findall('2', bToCheck))
            NoRemInCheck = NoRemInCheck1 + NoRemInCheck2

            if NoRemInCheck == EpochsPerWindow:
                startPoints.append(start)
                endPoints.append(end)
                EndNotFound = False
            else:
                end = end + 1

        start = end + 1

    # Add final window and limit to requested number
    startPoints.append(endPoints[-1] + 1)
    endPoints.append(len(b))

    if len(startPoints) > windows:
        startPoints = startPoints[:-1]
        endPoints = endPoints[:-1]

    startPoints[0] = bstart

    timePosition = []
    MADensityPoints = []

    # STEP 2: CALCULATE MA DENSITY FOR EACH WINDOW

    for (beg, end) in zip(startPoints, endPoints):
        # Calculate time position of window center
        timePosition.append(np.mean([beg, end]) * 4 / 3600)

        # Extract b-file segment for this window
        bPoints = b[beg:(end+1)]

        # Remove microarousals and get MA statistics
        [statesMACor, MASpans, PerMAs] = ElimMA(bPoints, 12)  # 12-second MA threshold

        # Count number of microarousals found
        MAs = MASpans.shape[0]

        # Calculate NREM time in this window (minutes)
        NoRemBouts1 = len(re.findall('n', bPoints))
        NoRemBouts2 = len(re.findall('2', bPoints))
        NoRemMin = (NoRemBouts1 + NoRemBouts2) * 4 / 60

        # Calculate MA density (MAs per minute of NREM)
        MADensityPoints.append(MAs / NoRemMin)

    # Convert to numpy arrays
    MADensityArray = np.array(MADensityPoints)
    timePositionArray = np.array(timePosition)

    return MADensityArray, timePositionArray


def CorrectMirrorAnimals(HypREMSDPair, HypMirrorPair, REMSDDur):
    """
    Correct the hypnogram of a mouse during a REM sleep deprivation mirror experiment.

    In mirror experiments, one mouse undergoes REM sleep deprivation while its pair
    serves as a control. This function corrects motor-induced awakenings in the
    control mouse's hypnogram using REM-to-wake (rw) transitions from the REMSD mouse.

    inputs:
    HypREMSDPair: hypnogram of the mouse undergoing REM sleep deprivation
    HypMirrorPair: hypnogram of the control (mirror) mouse
    REMSDDur: duration of REM sleep deprivation in hours

    outputs:
    HypMirrorPairCor: corrected hypnogram of the mirror mouse
    """

    Hyp1 = HypREMSDPair
    Hyp2 = HypMirrorPair

    # Convert REMSD duration to number of 4-second epochs
    REMSDHours = int(REMSDDur * 3600 / 4)

    # Extract REMSD period from first hypnogram
    bfileREMSD = Hyp1[0:REMSDHours]

    # STEP 1: FIND REM-TO-WAKE TRANSITIONS IN REMSD ANIMAL
    # These indicate motor interventions for REM sleep deprivation
    RWs = re.finditer('[r3][w1]', bfileREMSD)

    RWsSpans = []
    for idx, rw in enumerate(RWs):
        Beg = list(rw.span())[0]
        End = list(rw.span())[1] - 1
        RWsSpans.append([Beg, End])

    # Initialize arrays for visualization/analysis
    HypToPlot1 = np.zeros(len(Hyp1))
    HypToPlot2 = np.zeros(len(Hyp2))
    HypToPlotCor2 = np.zeros(len(Hyp2))

    # Create mutable copy of mirror hypnogram
    Hyp2Cor = list(Hyp2)

    # STEP 2: CORRECT WAKE PERIODS IN MIRROR ANIMAL
    # For each motor intervention time, mark corresponding wake periods as 'u' (unknown/artifact)

    for s in RWsSpans:
        # Check various patterns around motor intervention times
        # and mark wake periods as artifacts ('u')

        # Pattern 1: Non-wake followed by wake at intervention time
        if (not(Hyp2Cor[s[0]] == 'w' or Hyp2Cor[s[0]] == '1') and
              (Hyp2Cor[np.min([s[0]+1, len(Hyp2Cor)])] == 'w' or
               Hyp2Cor[np.min([s[0]+1, len(Hyp2Cor)])] == '1')):

            CharToCorrect = 'u'
            i = s[0] + 1

            if (s[0] + 1) < len(Hyp2Cor):
                # Mark consecutive wake epochs as artifacts
                while ((Hyp2Cor[np.min([i, len(Hyp2Cor)-1])] == 'w' or
                        Hyp2Cor[np.min([i, len(Hyp2Cor)-1])] == '1') and
                       i < len(Hyp2Cor)):
                    Hyp2Cor[i] = CharToCorrect
                    i += 1

        # Pattern 2: Wake period starts exactly at intervention time
        elif (not(Hyp2Cor[np.max([0, s[0]-1])] == 'w' or
                  Hyp2Cor[np.max([0, s[0]-1])] == '1') and
              (Hyp2Cor[s[0]] == 'w' or Hyp2Cor[s[0]] == '1')):

            CharToCorrect = 'u'
            i = s[0]

            # Mark consecutive wake epochs as artifacts
            while ((Hyp2Cor[np.min([i, len(Hyp2Cor)-1])] == 'w' or
                    Hyp2Cor[np.min([i, len(Hyp2Cor)-1])] == '1') and
                   i < len(Hyp2Cor)):
                Hyp2Cor[i] = CharToCorrect
                i += 1

    # Convert corrected list back to string
    Hyp2CorStr = ""
    Hyp2CorFinal = Hyp2CorStr.join(Hyp2Cor)
    HypMirrorPairCor = Hyp2CorFinal

    return HypMirrorPairCor


def cohen_d(x, y):
    """
    Calculate Cohen's d effect size between two groups.

    Cohen's d is a standardized measure of the difference between two group means,
    expressed in terms of pooled standard deviation units.

    inputs:
    x: first group data
    y: second group data

    outputs:
    Cohen's d effect size
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2  # Degrees of freedom

    # Calculate pooled standard deviation and effect size
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def findPosFromTimestamps(timeStamps, timeToLoc):
    """
    Find the position in a timestamp vector where a specific timepoint is located.

    This function finds the index of the timestamp closest to the target time.

    inputs:
    timeStamps: vector of recorded timestamps (in seconds)
    timeToLoc: time point to locate in timestamps (in seconds)

    outputs:
    pos: position/index in recorded timestamps
    """
    # Find minimum time difference and return corresponding index
    DiffTime = np.abs(timeStamps - timeToLoc)
    pos = np.where(DiffTime == np.min(DiffTime))[0][0]

    return pos


def binning(s, nbins=10):
    """
    Bin a signal into a specified number of bins and return mean values.

    This function divides a signal into equal-length bins and calculates
    the mean value within each bin for data reduction/summarization.

    inputs:
    s: the signal to be binned
    nbins: number of bins to compute (default: 10)

    outputs:
    sBinned: binned signal containing mean values
    """
    # Calculate length of each bin
    lengthPerBin = int(np.ceil(s.shape[0] / nbins))
    sBinned = []

    for b in range(nbins):
        if ((b + 1) * lengthPerBin <= s.shape[0]):
            # Full bin
            currentBin = s[(b * lengthPerBin):((b + 1) * lengthPerBin)]
            sBinned.append(np.mean(currentBin))
        else:
            # Last bin may be shorter
            currentBin = s[(b * lengthPerBin):]
            sBinned.append(np.mean(currentBin))

    return np.array(sBinned)


def MGT(v_Signal, fs, f1, f2, step):
    """
    Morlet Gabor Transform for time-frequency analysis.

    This function computes a time-frequency representation using Morlet wavelets,
    which are Gaussian-windowed complex exponentials providing good time-frequency
    localization.

    inputs:
    v_Signal: input signal for analysis
    fs: sampling frequency
    f1: minimum frequency for analysis
    f2: maximum frequency for analysis
    step: frequency step size

    outputs:
    m_Transform: time-frequency transform matrix
    """
    cycles = 4  # Number of cycles in the Morlet wavelet

    # Create frequency vector
    v_freq = np.arange(f1, f2 + step, step)
    s_Len = v_Signal.shape[0]
    s_HalfLen = int(np.floor(s_Len / 2) + 1)

    # Compute FFT of input signal
    v_YFFT = fft(v_Signal, v_Signal.shape[0])

    # Create angular frequency axis
    v_WAxis = (2 * np.pi / s_Len) * np.arange(0, s_Len)
    v_WAxis = v_WAxis * fs
    v_WAxisHalf = v_WAxis[0:(s_HalfLen)]

    m_Transform = []

    # For each frequency, create Morlet wavelet and convolve
    for i in np.arange(0, v_freq.shape[0]):
        s_ActFrq = v_freq[i]

        # Calculate wavelet duration based on number of cycles
        dtseg = cycles * (1 / s_ActFrq)

        # Create Gaussian window in frequency domain
        v_WinFFT = np.zeros([s_Len])
        v_WinFFT[0:s_HalfLen] = np.exp(-0.5 * np.power(v_WAxisHalf - 2 * np.pi * s_ActFrq, 2) * np.power(dtseg, 2))

        # Normalize the wavelet
        v_WinFFT = v_WinFFT * np.sqrt(s_Len) / np.linalg.norm(v_WinFFT, 2)

        # Convolve signal with wavelet (multiplication in frequency domain)
        m_Transform.append(ifft(v_YFFT * v_WinFFT) / np.sqrt(dtseg))

    m_Transform = np.array(m_Transform)
    return m_Transform


def nan_helper(y):
    """
    Helper function to handle indices and logical indices of NaNs.

    This utility function identifies NaN locations and provides indexing
    functions for interpolation operations.

    inputs:
    y: 1D numpy array with possible NaNs

    outputs:
    nans: logical indices of NaNs
    index: function to convert logical indices to numerical indices
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpNan(y):
    """
    Interpolate over NaN values in a 1D array using linear interpolation.

    This function fills NaN gaps in data using linear interpolation between
    valid neighboring points.

    inputs:
    y: 1D array with NaN values to be interpolated

    outputs:
    ycopy: array with NaN values replaced by interpolated values
    """
    ycopy = y.copy()

    # Identify NaN locations
    nans, x = nan_helper(ycopy)

    # Interpolate NaN values
    ycopy[nans] = np.interp(x(nans), x(~nans), ycopy[~nans])

    return ycopy


def BPM(emg, b, Fs, filtertrace=False):
    """
    Extract beats per minute (BPM/heart rate) from EMG signal.

    This function extracts heart rate information from bipolarized EMG signals
    by detecting cardiac artifacts and calculating inter-beat intervals.
    Translated from MATLAB version by Romain Cardis (2020).

    inputs:
    emg: bipolarized EMG signal
    b: scored b-file for sleep state information
    Fs: sampling frequency of EMG signal
    filtertrace: flag to apply low-pass filtering to output (default: False)

    outputs:
    bpm10: heart rate signal resampled to 10 Hz
    """

    # STEP 1: RESAMPLE TO 200 Hz IF NECESSARY
    if Fs == 1000:
        emg = signal.resample(emg, int(emg.shape[0] / 5))
        Fs = 200

    # STEP 2: HIGH-PASS FILTER TO ISOLATE CARDIAC ARTIFACTS
    # 25 Hz high-pass filter to remove low-frequency EMG and isolate heartbeat
    bf, af = signal.cheby2(6, 40, 25 / (Fs/2), btype='high', analog=False, output='ba')
    emg = signal.filtfilt(bf, af, emg)

    print('Squared Cor')

    # STEP 3: COMPUTE CARDIAC ACTIVITY SIGNAL
    # Square the derivative to emphasize sharp cardiac artifacts
    hr = np.abs(np.concatenate(([0], np.diff(emg))))**2

    # Create timestamp vector
    timestamps = np.arange(0, emg.shape[0]/Fs, 1/Fs)

    # STEP 4: NORMALIZE USING NREM PERIODS
    # Use NREM periods for normalization (stable cardiac activity)
    TimesOfb = np.arange(0, (len(b))*4, 4)

    # Find NREM triplets for normalization
    NREMbouts = re.finditer('(?=[nf][nf][nf])', b)

    PointsNorm = []

    # Extract heart rate values during NREM triplets
    for nrembout in NREMbouts:
        PointStart = int(nrembout.span()[0])
        PointEnd = int(nrembout.span()[0]) + 1

        PointStartSec = TimesOfb[PointStart]
        PointsEndSec = TimesOfb[PointEnd]

        PosStart = findPosFromTimestamps(timestamps, PointStartSec)
        PosEnd = findPosFromTimestamps(timestamps, PointsEndSec)

        PointsNorm.append(hr[PosStart:PosEnd])

    PointsNorm = np.array(PointsNorm)
    PointsNorm = PointsNorm.flatten()

    # Z-score normalization using NREM data
    hr = (hr - np.mean(PointsNorm)) / np.std(PointsNorm)

    # STEP 5: DETECT HEARTBEAT PEAKS
    # Find peaks with minimum height and distance constraints
    peaks, properties = signal.find_peaks(hr, height=0.3, distance=0.08*200)

    # Filter out extreme outliers
    peaksToKeep = []
    for i, peak in enumerate(peaks):
        if not(properties['peak_heights'][i] > 10):  # Remove very high peaks (artifacts)
            peaksToKeep.append(peak)

    # Convert peak indices to timestamps
    peaksToKeep = np.array(peaksToKeep) / Fs

    # STEP 6: CALCULATE BPM FROM INTER-BEAT INTERVALS
    bpm = 1. / np.diff(peaksToKeep) * 60  # Convert intervals to beats per minute

    # Remove physiologically implausible values
    toLow = bpm < 100  # Remove BPM < 100 (too low for mice)
    peaksToKeep = peaksToKeep[1::]

    peaksToKeep = np.delete(peaksToKeep, toLow)
    bpm = np.delete(bpm, toLow)

    # STEP 7: REMOVE SUDDEN JUMPS (ARTIFACTS)
    dbpm = np.concatenate(([0], np.abs(np.diff(bpm))))
    jum = dbpm > 200  # Mark sudden changes > 200 BPM as artifacts
    bpm[jum] = np.nan

    # Interpolate over artifacts
    bpmFinal = interpNan(bpm)

    # STEP 8: RESAMPLE TO 10 Hz
    x10 = np.linspace(0, (int(len(emg) / Fs) - 1), int((len(emg) / Fs)*10))
    bpm10 = np.interp(x10, peaksToKeep, bpmFinal)
    bpm10 = interpNan(bpm10)

    # Optional low-pass filtering
    if filtertrace:
        b = signal.firwin(100, 0.025 / 10, pass_zero='lowpass')
        bpm10 = signal.filtfilt(b, 1, bpm10)

    return bpm10


def Get_envelope_distance(x, y):
    """
    Calculate signal envelope using distance criterion for peak detection.

    This function computes upper and lower envelopes of a signal by finding
    peaks and troughs with a specified minimum distance constraint.

    inputs:
    x: input signal for envelope calculation
    y: distance criterion in number of points for peak separation

    outputs:
    peaks: indices of detected peaks
    troughs: indices of detected troughs
    Upper_envelope: upper envelope values
    Lower_envelope: lower envelope values
    """

    # Find peaks (maxima) with distance constraint
    peaks, _ = signal.find_peaks(x.squeeze(), height=0, distance=y)

    # Find troughs (minima) with distance constraint
    troughs, _ = signal.find_peaks(-x.squeeze(), distance=y)

    # Handle case with only one peak
    if peaks.shape[0] == 1:
        peaks, _ = signal.find_peaks(x.squeeze(), height=0)
        peaks = np.array([peaks[0], peaks[-1]])

    # Fit cubic splines to peaks and troughs for smooth envelopes
    u_p = interpolate.CubicSpline(peaks, x[peaks])
    l_p = interpolate.CubicSpline(troughs, x[troughs])

    # Generate envelope values for all time points
    Upper_envelope = [u_p(i) for i in range(x.shape[0])]
    Lower_envelope = [l_p(i) for i in range(x.shape[0])]

    return peaks, troughs, Upper_envelope, Lower_envelope


def Get_envelope(x):
    """
    Calculate signal envelope without distance constraints.

    This function computes upper and lower envelopes by finding all peaks
    and troughs above/below zero, then fitting smooth curves.

    inputs:
    x: input signal for envelope calculation

    outputs:
    peaks: indices of detected peaks
    troughs: indices of detected troughs
    Upper_envelope: upper envelope as numpy array
    Lower_envelope: lower envelope as numpy array
    """

    # Find all peaks above zero
    peaks, _ = signal.find_peaks(x.squeeze(), height=0)

    # Find all troughs (peaks of inverted signal)
    troughs, _ = signal.find_peaks(-x.squeeze())

    # Fit cubic splines for smooth envelopes
    u_p = interpolate.CubicSpline(peaks, x[peaks])
    l_p = interpolate.CubicSpline(troughs, x[troughs])

    # Generate envelope values
    Upper_envelope = [u_p(i) for i in range(x.shape[0])]
    Lower_envelope = [l_p(i) for i in range(x.shape[0])]

    return peaks, troughs, np.array(Upper_envelope), np.array(Lower_envelope)


def EqualTimeSpentInNrem(bstart, bfile, windows):
    """
    Split a recording into windows with equal time spent in NREM sleep.

    This function divides a sleep recording into a specified number of windows,
    each containing the same amount of NREM sleep time.

    inputs:
    bstart: start index of the b-file for calculation
    bfile: original scored b-file string
    windows: number of windows to create

    outputs:
    windowBorders: list of (start, end) tuples defining window boundaries in epochs
    """

    b = bfile
    startPoints = []
    endPoints = []

    # STEP 1: CALCULATE TOTAL NREM TIME
    # Count NREM epochs ('n' and '2' states)
    NoRemBouts1 = len(re.findall('n', b))
    NoRemBouts2 = len(re.findall('2', b))

    # Convert to hours
    NoRemAll = (NoRemBouts1 + NoRemBouts2) * 4 / 3600

    # STEP 2: CALCULATE NREM TIME PER WINDOW
    NoRemPerWindow = NoRemAll / windows
    EpochsPerWindow = int(np.floor(NoRemPerWindow * 3600 / 4))

    # STEP 3: FIND WINDOW BOUNDARIES
    # Split b-file into windows with equal NREM content
    end = 0
    start = 0

    while start <= len(b):
        EndNotFound = True

        while end <= len(b) and EndNotFound:
            # Check NREM content in current window candidate
            bToCheck = b[start:end]

            NoRemInCheck1 = len(re.findall('n', bToCheck))
            NoRemInCheck2 = len(re.findall('2', bToCheck))
            NoRemInCheck = NoRemInCheck1 + NoRemInCheck2

            if NoRemInCheck == EpochsPerWindow:
                # Found window with correct NREM amount
                startPoints.append(start)
                endPoints.append(end)
                EndNotFound = False
            else:
                end = end + 1

        start = end + 1

    # Add final window
    startPoints.append(endPoints[-1] + 1)
    endPoints.append(len(b))

    # Limit to requested number of windows
    if len(startPoints) > windows:
        startPoints = startPoints[:-1]
        endPoints = endPoints[:-1]

    # Set first window start point
    startPoints[0] = bstart

    # Create list of window border tuples
    windowBorders = zip(startPoints, endPoints)

    return list(windowBorders) == '1') and
                   i < len(Hyp2Cor)):
                Hyp2Cor[i] = CharToCorrect
                i += 1

        # Pattern 3: Wake period starts 1 epoch before intervention
        elif (not(Hyp2Cor[np.max([s[0]-2, 0])] == 'w' or
                  Hyp2Cor[np.max([s[0]-2, 0])] == '1') and
              (Hyp2Cor[np.max([s[0]-1, 0])] == 'w' or
               Hyp2Cor[np.max([s[0]-1, 0])] == '1') and
              (Hyp2Cor[np.max([s[0], 0])] == 'w' or
               Hyp2Cor[np.max([s[0], 0])] == '1')):

            CharToCorrect = 'u'
            i = s[0] - 1

            if ((s[0]-1) >= 0):
                # Mark consecutive wake epochs as artifacts
                while ((Hyp2Cor[np.min([i, len(Hyp2Cor)-1])] == 'w' or
                        Hyp2Cor[np.min([i, len(Hyp2Cor)-1])] == '1') and
                       i < len(Hyp2Cor)):
                    Hyp2Cor[i] = CharToCorrect
                    i += 1

        # Pattern 4: Wake period starts 2-3 epochs after intervention
        elif (not(Hyp2Cor[np.min([s[0]+2, len(Hyp2Cor)-1])] == 'w' or
                  Hyp2Cor[np.min([s[0]+2, len(Hyp2Cor)-1])] == '1') and
              (Hyp2Cor[np.min([s[0]+3, len(Hyp2Cor)-1])] == 'w' or
               Hyp2Cor[np.min([s[0]+3, len(Hyp2Cor)-1])] == '1')):

            CharToCorrect = 'u'

            # First mark intervention period as wake
            Hyp2Cor[s[0]:np.min([s[0]+3, len(Hyp2Cor)])] = list('w'*3)

            i = s[0]
            # Then mark all as artifacts
            while ((Hyp2Cor[np.min([i, len(Hyp2Cor)-1])] == 'w' or
                    Hyp2Cor[np.min([i, len(Hyp2Cor)-1])] == '1') and
                   i < len(Hyp2Cor)):
                Hyp2Cor[i] = CharToCorrect
                i += 1

        # Pattern 5: Wake period starts 1 epoch after intervention
        elif (not(Hyp2Cor[np.min([s[0]+1, len(Hyp2Cor)-1])] == 'w' or
                  Hyp2Cor[np.min([s[0]+1, len(Hyp2Cor)-1])] == '1') and
              (Hyp2Cor[np.min([s[0]+2, len(Hyp2Cor)-1])] == 'w' or
               Hyp2Cor[np.min([s[0]+2, len(Hyp2Cor)-1])] == '1')):

            CharToCorrect = 'u'

            # First mark intervention period as wake
            Hyp2Cor[s[0]:np.min([s[0]+2, len(Hyp2Cor)])] = list('w'*2)

            i = s[0]
            # Then mark all as artifacts
            while ((Hyp2Cor[np.min([i, len(Hyp2Cor)-1])] == 'w' or
                    Hyp2Cor[np.min([i, len(Hyp2Cor)-1])]
