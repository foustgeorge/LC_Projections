"""
PeakDetection.py

Locus Coeruleus (LC) peak detection algorithm for fiber photometry signals.

This module implements the peak detection algorithm described in Osorio-Forero,
Foustoukos et al. 2024 for identifying LC activity peaks in calcium imaging data
during sleep-wake transitions.

Original MATLAB version by Dr. Alejandro Osorio-Forero
Python adaptation by Dr. Georgios Foustoukos and Paola Milanese (May 2024)

Author: Dr. Georgios Foustoukos, Paola Milanese
Date: May 2024
"""

import numpy as np
import re
from scipy import signal
from scipy import interpolate
from scipy import optimize


def elim_artefacts(b):
    """
    Convert numerical sleep state artifacts to standard letter notation.

    This function standardizes sleep state scoring by converting numerical
    artifacts (1,2,3) to standard letter notation used in sleep research.

    Parameters
    ----------
    b : str
        Sleep state string with numerical artifacts (1,2,3)

    Returns
    -------
    str
        Clean sleep state string with standard notation:
        - '1' -> 'w' (Wake)
        - '2' -> 'n' (NREM sleep)
        - '3' -> 'r' (REM sleep)

    Notes
    -----
    This preprocessing step ensures consistent sleep state notation
    throughout the analysis pipeline.
    """
    b_noArtefacts = b.replace('1', 'w')
    b_noArtefacts = b_noArtefacts.replace('2', 'n')
    b_noArtefacts = b_noArtefacts.replace('3', 'r')

    return b_noArtefacts


def elim_MA(b):
    """
    Identify and eliminate microarousals (MA) from sleep state data.

    Microarousals are brief (<12s) wake periods during sleep that can
    confound analysis. This function identifies MAs using regex patterns
    and replaces them with NREM sleep state.

    Parameters
    ----------
    b : str
        Sleep state string (w=wake, n=NREM, r=REM)

    Returns
    -------
    ma_index : np.ndarray
        Indices where microarousals were detected
    b_noMA : str
        Sleep state string with microarousals replaced by NREM ('n')

    Notes
    -----
    Detects three types of microarousals:
    - Type 1: 'nwn' (4s MA: NREM-Wake-NREM)
    - Type 2: 'nwwn' (8s MA: NREM-Wake-Wake-NREM)
    - Type 3: 'nwwwn' (12s MA: NREM-Wake-Wake-Wake-NREM)

    Each character represents a 4-second scoring epoch.
    """
    # STEP 1: IDENTIFY MICROAROUSALS USING REGEX PATTERNS

    # Find 4-second microarousals (n-w-n pattern)
    ma_1 = re.finditer('(?=nwn)', b)

    # Find 8-second microarousals (n-w-w-n pattern)
    ma_2 = re.finditer('(?=nwwn)', b)

    # Find 12-second microarousals (n-w-w-w-n pattern)
    ma_3 = re.finditer('(?=nwwwn)', b)

    # STEP 2: EXTRACT MA POSITIONS

    # Extract start positions for 4s MAs (middle wake epoch)
    array_ma1 = []
    for ma in ma_1:
        array_ma1.append(ma.start() + 1)  # Position of wake epoch
    array_ma1 = np.array(array_ma1)

    # Extract start positions for 8s MAs (first wake epoch)
    array_ma2 = []
    for ma in ma_2:
        array_ma2.append(ma.start() + 1)  # Position of first wake epoch
    array_ma2 = np.array(array_ma2)

    # Extract start positions for 12s MAs (first wake epoch)
    array_ma3 = []
    for ma in ma_3:
        array_ma3.append(ma.start() + 1)  # Position of first wake epoch
    array_ma3 = np.array(array_ma3)

    # Combine all MA indices and sort
    ma_index = np.concatenate((array_ma1, array_ma2, array_ma3))
    ma_index = sorted(ma_index)

    # STEP 3: REPLACE MICROAROUSALS WITH NREM
    b_noMA = b
    b_noMA = b_noMA.replace('nwn', 'nnn')      # Replace 4s MA
    b_noMA = b_noMA.replace('nwwn', 'nnnn')    # Replace 8s MA
    b_noMA = b_noMA.replace('nwwwn', 'nnnnn')  # Replace 12s MA

    return ma_index, b_noMA


def b_num(b, time):
    """
    Convert sleep state string to numerical format matched to signal sampling rate.

    This function creates a numerical sleep state array that matches the
    sampling frequency of fiber photometry acquisition, enabling precise
    temporal alignment between sleep states and calcium signals.

    Parameters
    ----------
    b : str
        Sleep state string where each character represents 4 seconds
    time : np.ndarray
        Timestamps of fiber photometry acquisition (seconds)

    Returns
    -------
    np.ndarray
        Numerical sleep state array matching photometry sampling rate:
        - 1: REM sleep
        - 2: NREM sleep
        - 3: Wake

    Notes
    -----
    Each character in the sleep state string represents a 4-second epoch.
    The function finds the closest b-file timepoint for each photometry
    sample and assigns the corresponding numerical sleep state.
    """
    # Initialize numerical array matching photometry length
    b_numerized = np.zeros(len(time), dtype=int)

    # Create time vector for sleep scoring (4-second epochs)
    timeInb = np.arange(0, (len(b)-1)*4, 4)

    # STEP 1: MATCH EACH PHOTOMETRY TIMEPOINT TO SLEEP STATE
    for i, t in enumerate(time):
        # Find closest sleep scoring timepoint
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


def Get_envelope_distance(x, y):
    """
    Calculate signal envelope using distance-constrained peak detection.

    Computes upper and lower envelopes of a signal by finding peaks and troughs
    with a minimum separation distance, then fitting cubic splines for smooth
    envelope curves.

    Parameters
    ----------
    x : np.ndarray
        Input signal for envelope calculation
    y : int
        Minimum distance between peaks/troughs (number of samples)

    Returns
    -------
    tuple
        peaks : np.ndarray
            Indices of detected peaks
        troughs : np.ndarray
            Indices of detected troughs
        Upper_envelope : list
            Upper envelope values for each sample
        Lower_envelope : list
            Lower envelope values for each sample

    Notes
    -----
    Uses cubic spline interpolation between detected peaks/troughs to create
    smooth envelope curves. Distance constraint prevents detection of spurious
    peaks in noisy signals.
    """
    # STEP 1: DETECT PEAKS AND TROUGHS WITH DISTANCE CONSTRAINT

    # Find peaks (local maxima) above zero with minimum separation
    peaks, _ = signal.find_peaks(x.squeeze(), height=0, distance=y)

    # Find troughs (local minima) with minimum separation
    troughs, _ = signal.find_peaks(-x.squeeze(), distance=y)

    # STEP 2: HANDLE EDGE CASES WITH INSUFFICIENT PEAKS/TROUGHS

    # If only one peak found, use first and last peaks without distance constraint
    if peaks.shape[0] == 1:
        peaks, _ = signal.find_peaks(x.squeeze(), height=0)
        peaks = np.array([peaks[0], peaks[-1]])

    # If only one trough found, use first and last troughs
    if troughs.shape[0] == 1:
        troughs, _ = signal.find_peaks(-x.squeeze())
        troughs = np.array([troughs[0], troughs[-1]])

    # STEP 3: FIT CUBIC SPLINES TO CREATE SMOOTH ENVELOPES

    # Fit cubic spline to peaks for upper envelope
    u_p = interpolate.CubicSpline(peaks, x[peaks])

    # Fit cubic spline to troughs for lower envelope
    l_p = interpolate.CubicSpline(troughs, x[troughs])

    # Generate envelope values for all sample points
    Upper_envelope = [u_p(i) for i in range(x.shape[0])]
    Lower_envelope = [l_p(i) for i in range(x.shape[0])]

    return peaks, troughs, Upper_envelope, Lower_envelope


def Get_envelope(x):
    """
    Calculate signal envelope without distance constraints.

    Computes upper and lower envelopes by finding all peaks above zero
    and all troughs, then fitting cubic splines for smooth curves.

    Parameters
    ----------
    x : np.ndarray
        Input signal for envelope calculation

    Returns
    -------
    tuple
        peaks : np.ndarray
            Indices of all detected peaks
        troughs : np.ndarray
            Indices of all detected troughs
        Upper_envelope : list
            Upper envelope values
        Lower_envelope : list
            Lower envelope values

    Notes
    -----
    Less restrictive than Get_envelope_distance() - finds all peaks/troughs
    without minimum separation constraints.
    """
    # Find all peaks above zero
    peaks, _ = signal.find_peaks(x.squeeze(), height=0)

    # Find all troughs (peaks of inverted signal)
    troughs, _ = signal.find_peaks(-x.squeeze())

    # Fit cubic splines and generate envelopes
    u_p = interpolate.CubicSpline(peaks, x[peaks])
    l_p = interpolate.CubicSpline(troughs, x[troughs])
    Upper_envelope = [u_p(i) for i in range(x.shape[0])]
    Lower_envelope = [l_p(i) for i in range(x.shape[0])]

    return peaks, troughs, Upper_envelope, Lower_envelope


def Expfunc(x, a, b):
    """
    Exponential function for curve fitting.

    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Decay/growth rate parameter

    Returns
    -------
    np.ndarray
        Exponential function values: a * exp(b * x)
    """
    return a * np.exp(b * x)


def findPosFromTimestamps(timeStamps, timeToLoc):
    """
    Find the index position of a specific timepoint in a timestamp array.

    Locates the timestamp closest to the target time and returns its index
    position in the array.

    Parameters
    ----------
    timeStamps : np.ndarray
        Vector of recorded timestamps (seconds)
    timeToLoc : float
        Target time to locate in timestamps (seconds)

    Returns
    -------
    int
        Index position of closest timestamp

    Notes
    -----
    Uses absolute difference to find the closest match, useful for aligning
    different sampling rates or finding events in continuous data.
    """
    # Calculate absolute time differences
    DiffTime = np.abs(timeStamps - timeToLoc)

    # Find index of minimum difference
    pos = np.where(DiffTime == np.min(DiffTime))[0][0]

    return pos


def func(x, a, b):
    """
    Exponential decay function for curve fitting optimization.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (typically time)
    a : float
        Initial amplitude
    b : float
        Decay rate constant

    Returns
    -------
    np.ndarray
        Exponential decay: a * exp(b * x)

    Notes
    -----
    Used with scipy.optimize.curve_fit to determine LC activity decay
    time constants during NREM sleep transitions.
    """
    return a * np.exp(b * x)


def DynamicRange(signal):
    """
    Calculate the dynamic range of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Dynamic range (maximum - minimum)

    Notes
    -----
    Simple metric for signal amplitude characterization.
    """
    return np.max(signal) - np.min(signal)


def PeakDetection(dff, b_file, time):
    """
    Extract LC peak activity from fiber photometry signals using the algorithm
    described in Osorio-Forero, Foustoukos et al. 2024.

    This function implements a comprehensive peak detection pipeline specifically
    designed for identifying locus coeruleus (LC) calcium transients during
    sleep-wake transitions. The algorithm includes multiple filtering steps,
    envelope-based baseline correction, and physiologically-informed peak
    validation criteria.

    Parameters
    ----------
    dff : np.ndarray
        Delta F/F calcium signal from LC fiber photometry
    b_file : str
        Sleep state string with 4-second resolution scoring:
        - 'w': Wake
        - 'n': NREM sleep
        - 'r': REM sleep
    time : np.ndarray
        Timestamps of dF/F acquisition (seconds)

    Returns
    -------
    tuple
        peaksLCFreq : np.ndarray
            Indices of detected LC peaks (after prominence filtering)
        peaks_dff3_index : np.ndarray
            Final validated peak indices (after all filtering steps)
        PeaksFlag : list
            Binary flags indicating microarousal association:
            - 0: Peak not associated with microarousal
            - 1: Peak associated with microarousal (within 5s)

    Notes
    -----
    Algorithm Steps:
    1. Preprocessing: Remove artifacts and microarousals
    2. Signal filtering: Apply dual low-pass filtering (0.1 Hz and 0.5 Hz)
    3. Baseline correction: Use lower envelope to flatten signal
    4. Peak detection: Multi-step prominence and amplitude filtering
    5. Physiological validation: Remove peaks near wake transitions
    6. Sliding window deduplication: Keep only highest peaks in 20s windows
    7. Microarousal classification: Flag peaks associated with MAs

    The algorithm is optimized for detecting physiologically relevant LC
    activity while minimizing false positives from movement artifacts and
    non-specific fluorescence changes.

    References
    ----------
    Osorio-Forero, Foustoukos et al. 2024. [Full citation needed]
    """
    # STEP 1: CALCULATE SAMPLING FREQUENCY AND INITIALIZE PARAMETERS

    sampling_freq = int(np.round(1 / np.mean(np.diff(time))))
    wakelengthbeforeNREM = 60  # Minimum wake duration before NREM (seconds)

    # STEP 2: PREPROCESSING - REMOVE ARTIFACTS AND MICROAROUSALS

    # Convert numerical artifacts to standard notation
    b_noArtefacts = elim_artefacts(b_file)

    # Identify and remove microarousals, get clean sleep state data
    ma_index, b_noMA_noArt = elim_MA(b_noArtefacts)

    # Convert to numerical format matching signal sampling rate
    b_numerized = b_num(b_noMA_noArt, time)

    # STEP 3: ARTIFACT MITIGATION AT SIGNAL BOUNDARIES
    # Set first and last seconds to wake state to avoid edge artifacts

    b_numerized[-(1*sampling_freq):] = np.ones((1*sampling_freq), dtype='int') * 3
    b_numerized[0:(1*sampling_freq)] = np.ones((1*sampling_freq), dtype='int') * 3
    dff[-(2*sampling_freq):] = np.ones(2*sampling_freq, dtype='int') * 3
    dff[0:2*sampling_freq] = np.ones(2*sampling_freq, dtype='int') * 3

    # STEP 4: DUAL-PASS LOW-PASS FILTERING
    # Apply two different cutoff frequencies for different analysis purposes

    # High cutoff (0.5 Hz): Preserves faster LC transients
    high_lowpass = signal.firwin(1000, 0.5 / (sampling_freq / 2), pass_zero='lowpass')

    # Low cutoff (0.1 Hz): For envelope calculation and slow trend analysis
    low_lowpass = signal.firwin(1000, 0.1 / (sampling_freq / 2), pass_zero='lowpass')

    # Apply filters using zero-phase filtering
    dff_high = signal.filtfilt(high_lowpass, 1, dff)
    dff_low = signal.filtfilt(low_lowpass, 1, dff)

    # STEP 5: ENVELOPE-BASED BASELINE CORRECTION
    # Calculate signal envelope to remove slow baseline drift

    # Compute envelope with 96-second distance constraint between peaks
    peaks, troughs, dff_upper_enveloppe, dff_lower_enveloppe = Get_envelope_distance(
        dff_low, 96 * sampling_freq)

    # Flatten signal by subtracting lower envelope (baseline correction)
    signal_dff = (dff_high - dff_lower_enveloppe)

    # STEP 6: RESTRICT ANALYSIS TO NREM PERIODS ONLY
    # Set non-NREM periods to NaN to focus on NREM-specific LC activity
    signal_dff[b_numerized != 2] = np.nan

    # STEP 7: INITIAL PEAK DETECTION
    # Find all potential peaks in the baseline-corrected signal

    peaks_dff1_index, _ = signal.find_peaks(signal_dff)
    peaks1 = dff[peaks_dff1_index]
    prominences1 = signal.peak_prominences(signal_dff, peaks_dff1_index)[0]

    # STEP 8: PROMINENCE-BASED PEAK FILTERING
    # Remove low-prominence peaks using 60th percentile threshold

    peaks_dff2_index, _ = signal.find_peaks(signal_dff, prominence=np.percentile(prominences1, 60))
    peaks2 = dff[peaks_dff2_index]
    prominences2 = signal.peak_prominences(signal_dff, peaks_dff2_index)[0]

    # STEP 9: AMPLITUDE AND PROMINENCE DUAL-CRITERIA FILTERING
    # Apply stricter criteria: 25% of max prominence AND 20% of max amplitude

    todelete = np.zeros(len(peaks2))
    for idx_peak in range(len(peaks2)):
        # Mark for deletion if peak fails both prominence and amplitude criteria
        if ((prominences2[idx_peak] < 0.25 * np.nanmax(signal_dff[peaks_dff2_index])) &
            (signal_dff[peaks_dff2_index[idx_peak]] < np.nanmax(signal_dff) * 0.2)):
            todelete[idx_peak] = 1

    # Remove peaks that failed filtering criteria
    peaks3 = np.delete(peaks2, todelete == 1)
    prominences3 = np.delete(prominences2, todelete == 1)
    peaks_dff2_index = np.delete(peaks_dff2_index, todelete == 1)

    # Store intermediate result (after prominence filtering)
    peaksLCFreq = peaks_dff2_index

    # STEP 10: IDENTIFY LONG NREM BOUTS AFTER EXTENDED WAKE
    # Find NREM periods (≥90s) following long wake periods (≥60s)
    # Pattern: 15 epochs wake (60s) + ≥24 epochs NREM (≥96s)

    longNREM_re = re.finditer('w{15}n{24,}', b_noMA_noArt)
    longNREM_start = []
    longNREM_end = []

    for long_nonrem in longNREM_re:
        span = long_nonrem.span()
        longNREM_start.append(span[0])
        longNREM_end.append(span[1])

    longNREM_start = np.array(longNREM_start)
    longNREM_end = np.array(longNREM_end)

    # Remove bouts at recording edges (>99% of total length)
    if len(longNREM_end) > 0 and longNREM_end[-1] > len(b_noMA_noArt) * 0.99:
        longNREM_start = np.delete(longNREM_start, -1)
        longNREM_end = np.delete(longNREM_end, -1)

    # STEP 11: EXPONENTIAL DECAY ANALYSIS FOR TAU CALCULATION
    # Analyze LC activity decay during wake-to-NREM transitions

    # Use highly filtered signal with baseline correction
    dff_low_corrected = dff_low - np.min(dff_low)

    # Extract wake-to-NREM transition segments
    waketoNREMbouts_dff = []
    timesWaketoNREbouts = []
    currentbout = []

    for idx_Bout in range(len(longNREM_start)):
        # Convert epoch indices to signal timestamps
        timeStart = findPosFromTimestamps(time, longNREM_start[idx_Bout] * 4)
        timeStop = findPosFromTimestamps(time, longNREM_end[idx_Bout] * 4)

        # Extract and normalize signal segment
        currentbout = dff_low_corrected[timeStart:timeStop]
        currentbout = currentbout - np.min(currentbout)
        waketoNREMbouts_dff.append(currentbout)

        # Store corresponding timestamps
        timesWaketoNREbouts.append(time[timeStart:timeStop])

    # STEP 12: FIT EXPONENTIAL DECAY TO CALCULATE TIME CONSTANTS
    # Compute tau (time constant) for each wake-to-NREM transition

    tau = np.zeros(len(longNREM_start))
    for bout in range(len(longNREM_start)):
        # Create relative time vector starting from bout onset
        time_vec = timesWaketoNREbouts[bout] - timesWaketoNREbouts[bout][0]

        # Calculate envelope with 96-second distance constraint
        peaks, troughs, Upper_envelope, Lower_envelope = Get_envelope_distance(
            waketoNREMbouts_dff[bout], 96 * sampling_freq)

        Lower_envelope = np.array(Lower_envelope)

        # Extract NREM period (after initial wake period)
        timeNREMAfterLongWake = time_vec[time_vec > wakelengthbeforeNREM]
        EnvDffNREMAfterLongWake = Lower_envelope[time_vec > wakelengthbeforeNREM]

        # Fit exponential decay function to lower envelope
        try:
            a, b = optimize.curve_fit(func, timeNREMAfterLongWake,
                                    EnvDffNREMAfterLongWake, p0=(1, 1e-6))
            # Calculate time constant: tau = -1/decay_rate
            tau[bout] = -1 / (a[1])
        except:
            # Handle fitting failures
            tau[bout] = np.nan

    # STEP 13: OUTLIER REMOVAL AND MEAN TAU CALCULATION
    # Remove physiologically implausible tau values

    todelete = np.zeros(len(tau))
    for idx_bout in range(len(tau)):
        # Remove negative or excessively large tau values (>1000s)
        if np.isnan(tau[idx_bout]) or tau[idx_bout] < 0 or tau[idx_bout] > 1000:
            todelete[idx_bout] = 1

    tau = np.delete(tau, todelete == 1)

    # Calculate average time constant if valid taus exist
    if len(tau) > 0:
        av_tau = np.mean(tau)
    else:
        av_tau = 300  # Default fallback value (5 minutes)

    # STEP 14: PHYSIOLOGICAL PEAK VALIDATION
    # Remove peaks occurring within 1.5*tau of preceding wake events

    time_vec2 = time - time[0]  # Relative time vector

    count = 0
    todelete = np.zeros(len(peaks_dff2_index))

    for idxpeak in peaks_dff2_index:
        # Create boolean mask for time window before peak
        current = np.zeros(dff.shape[0])
        time_window_mask = np.logical_and(
            time_vec2 < time_vec2[idxpeak],
            time_vec2 > (time_vec2[idxpeak] - av_tau * 1.5))
        current[time_window_mask] = 1

        # Check if any wake states occurred in the time window
        if any(b_numerized[current == 1] > 2):  # Wake state = 3
            todelete[count] = 1
        count += 1

    # Remove peaks that failed physiological validation
    peaks_dff2_index = np.delete(peaks_dff2_index, todelete == 1)

    # STEP 15: SLIDING WINDOW DEDUPLICATION
    # In each 20-second window, keep only the peak with highest amplitude

    IndexForDelete = np.array(list(range(len(peaks_dff2_index))))
    peaks_dff3_index = peaks_dff2_index
    SlidingWindow = 20 * sampling_freq  # 20-second window

    toDelete = np.zeros(len(peaks_dff3_index))
    toDelete = np.array(toDelete, dtype='bool')

    # Slide window across entire signal
    for idx_epoch in range(len(dff) - SlidingWindow + 1):
        # Find peaks within current window
        RangeBoolean = np.logical_and(peaks_dff3_index >= idx_epoch,
                                     peaks_dff3_index < idx_epoch + SlidingWindow)

        # Exclude already marked peaks
        for b in range(len(RangeBoolean)):
            if toDelete[b]:
                RangeBoolean[b] = False

        CurrentPeaks = peaks_dff3_index[RangeBoolean]
        CurrentIndexes = IndexForDelete[RangeBoolean]

        # If multiple peaks in window, keep only the highest
        if len(CurrentPeaks) > 1:
            PeaksAmplitude = dff[CurrentPeaks]
            HighestPeak = np.argmax(PeaksAmplitude)

            # Mark all peaks except highest for deletion
            EliminatePeaks = np.ones(len(CurrentPeaks))
            EliminatePeaks = np.array(EliminatePeaks, dtype='bool')
            EliminatePeaks[HighestPeak] = False

            toDelete[CurrentIndexes[EliminatePeaks]] = True

    # Remove duplicated peaks
    peaks_dff3_index = np.delete(peaks_dff2_index, toDelete == 1)

    # STEP 16: MICROAROUSAL ASSOCIATION CLASSIFICATION
    # Classify peaks based on temporal association with microarousals

    # Convert MA indices to timestamps
    timeInb = np.arange(0, (len(b_file)) * 4, 4)
    MATimes = timeInb[ma_index]
    PeakTimes = time[peaks_dff3_index]
    PeaksFlag = []

    # For each peak, check if it occurs within 5 seconds of any MA
    for p in PeakTimes:
        TimeDiff = np.abs(MATimes - p)

        if np.sum(TimeDiff < 5) > 0:
            PeaksFlag.append(1)  # Peak associated with MA
        else:
            PeaksFlag.append(0)  # Peak not associated with MA

    return peaksLCFreq, peaks_dff3_index, PeaksFlag
