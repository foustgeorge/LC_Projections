#packages needed
import numpy as np
from scipy import signal
import re
from Schema.UsefulFunctions import findPosFromTimestamps

def baseline(dff, timestamps, window, b):
    """
    This function computes the baseline of a calcium signal using the
    5th percentile method on a 1Hz lowpass filtered signal, then uses
    sleep state data (b file) to fit a line using only NREM and REM
    bouts while excluding wake periods.

    Inputs:
    dff: input calcium dF/F signal
    timestamps: timestamps of the acquisition
    window: time window (in seconds) around each point for percentile calculation
    b: sleep state string where each character represents 4-second epochs
       (typical format: 'w'=wake, 'n'=NREM, 'r'=REM, '2'=stage 2, etc.)

    Outputs:
    fitbas: polynomial-fitted baseline using only NREM/REM periods
    """

    # Store original signal and timestamps
    signaldff = dff
    signalTime = timestamps

    # Calculate sampling frequency from timestamp differences
    fs = int(np.round(1 / np.mean(np.diff(timestamps))))

    # STEP 1: FILTER THE SIGNAL AT 1Hz TO REMOVE CALCIUM ACTIVITY

    # Design a low-pass FIR filter with cutoff at 0.01 normalized frequency
    # This creates a 1000-tap filter with cutoff at ~1Hz
    lpfilt = signal.firwin(1000, 0.01 / (fs / 2), pass_zero='lowpass')

    # Apply zero-phase filtering to avoid phase distortion
    filtsing = signal.filtfilt(lpfilt, -1, signaldff)

    # STEP 2: CALCULATE SLIDING WINDOW 5th PERCENTILE BASELINE

    # Initialize baseline array
    baseline = np.zeros_like(filtsing)

    print('Baseline Calculation started!')

    # For each time point, calculate 5th percentile in surrounding window
    for i_timepoint in range(len(filtsing)):
        # Define window boundaries (convert window from seconds to samples)
        start_idx = np.max([0, i_timepoint - window * fs])
        end_idx = np.min([len(filtsing), i_timepoint + window * fs])

        # Extract signal within window
        windowSignal = filtsing[start_idx:end_idx]

        # Calculate 5th percentile as baseline estimate
        baseline[i_timepoint] = np.percentile(windowSignal, 5)

    print('Baseline Calculated!')

    # STEP 3: PARSE SLEEP STATE DATA TO IDENTIFY NON-WAKE PERIODS

    # Create time array for b-file (each character = 4 seconds)
    TimesOfb = np.arange(0, len(b) * 4, 4)

    # Regular expression to find sequences of non-wake states
    # '[n2]+[^w]*' matches: one or more 'n' or '2', followed by any non-'w' characters
    exp = '[n2]+[^w]*'
    NoWake = re.finditer(exp, b)

    # Extract start and stop indices of non-wake periods
    NoWakeStart = []
    NoWakeStop = []

    for nowake in NoWake:
        span = nowake.span()
        NoWakeStart.append(span[0])
        NoWakeStop.append(span[1])

    # STEP 4: EXTRACT BASELINE VALUES DURING NON-WAKE PERIODS

    # Initialize lists to store baseline data for fitting
    points = []          # Time points for fitting
    baselineAprox = []   # Corresponding baseline values

    # Variables for interpolation intervals
    interpIntervals = 0
    NREMBaselineSignal = []
    NREMBaselinePoints = []
    NREMBaselineTimesOfB = []

    # Variables to track interpolation boundaries
    NREMBaselineInterTimeStartPoint = []
    NREMBaselineInterTimeStartPointBfile = []
    NREMBaselineInterTimeEndPoint = []
    NREMBaselineInterTimeEndPointBfile = []

    # Process each non-wake period
    for i, s in enumerate(zip(NoWakeStart, NoWakeStop)):
        # Convert b-file indices to actual time
        StartIndexToKeep = TimesOfb[s[0]]
        StopIndexToKeep = TimesOfb[s[1]-1] + 4

        # Find corresponding indices in the DFF signal
        StartIndexDff = findPosFromTimestamps(signalTime, StartIndexToKeep)
        StopIndexDff = findPosFromTimestamps(signalTime, StopIndexToKeep)

        # Extract baseline data for this non-wake period
        NREMBaselineSignal.append(baseline[StartIndexDff:StopIndexDff])
        NREMBaselinePoints.append(signalTime[StartIndexDff:StopIndexDff])
        NREMBaselineTimesOfB.append(np.arange(TimesOfb[s[0]], TimesOfb[s[1]-1] + 4, 4))

        # Add to fitting data
        points.append(signalTime[StartIndexDff:StopIndexDff])
        baselineAprox.append(baseline[StartIndexDff:StopIndexDff])

        # Skip if this period starts at the beginning
        if NoWakeStart[i] == 0:
            continue

        # Handle interpolation intervals between non-wake periods
        if i == 0 and not(NoWakeStart[i] == 0):
            # First period doesn't start at beginning - mark start as NaN
            interpIntervals += 1
            NREMBaselineInterTimeStartPoint.append(np.nan)
            NREMBaselineInterTimeStartPointBfile.append(np.nan)
            EndPoint = findPosFromTimestamps(signalTime, StartIndexToKeep)
            NREMBaselineInterTimeEndPoint.append(signalTime[EndPoint])
            NREMBaselineInterTimeEndPointBfile.append(TimesOfb[s[0]])
            continue

        # Handle gaps between consecutive non-wake periods
        interpIntervals += 1
        StartPoint = findPosFromTimestamps(signalTime, TimesOfb[NoWakeStop[i-1]])

        NREMBaselineInterTimeStartPoint.append(signalTime[StartPoint-1])
        NREMBaselineInterTimeStartPointBfile.append(TimesOfb[NoWakeStop[i-1]])

        Endoint = findPosFromTimestamps(signalTime, TimesOfb[NoWakeStart[i]])

        NREMBaselineInterTimeEndPoint.append(signalTime[Endoint])
        NREMBaselineInterTimeEndPointBfile.append(TimesOfb[NoWakeStart[i]])

        # Handle case where last period doesn't end at the end of recording
        if i == (len(NoWakeStart) - 1) and not(NoWakeStop[i] == len(b)):
            interpIntervals += 1
            StartPoint = findPosFromTimestamps(signalTime, TimesOfb[NoWakeStop[i]])

            NREMBaselineInterTimeStartPoint.append(signalTime[StartPoint-1])
            NREMBaselineInterTimeStartPointBfile.append(TimesOfb[NoWakeStop[i]])

            NREMBaselineInterTimeEndPoint.append(np.nan)
            NREMBaselineInterTimeEndPointBfile.append(np.nan)

    # STEP 5: HANDLE INTERPOLATION INTERVALS (WAKE PERIODS)

    InterLinesSignal = []
    InterLinesPoints = []

    # For each interpolation interval, add NaN values (excluded from fitting)
    for i, p in enumerate(zip(NREMBaselineInterTimeStartPoint, NREMBaselineInterTimeEndPoint)):

        if np.isnan(p[0]):
            # Start of recording until first non-wake period
            time = p[1]
            Pos = findPosFromTimestamps(signalTime, time)

            InterLinesSignal.append(np.ones([Pos]) * np.nan)
            InterLinesPoints.append(signalTime[0:Pos])

            points.append(signalTime[0:Pos])
            baselineAprox.append(np.ones([Pos]) * np.nan)

        elif np.isnan(p[1]):
            # Last non-wake period until end of recording
            time = p[0]
            Pos = findPosFromTimestamps(signalTime, time)

            InterLinesSignal.append(np.ones([max(0, len(baseline)-Pos-1)]) * np.nan)
            InterLinesPoints.append(signalTime[(Pos+1)::])

            points.append(signalTime[(Pos+1)::])
            baselineAprox.append(np.ones([max(0, len(baseline)-Pos-1)]) * np.nan)

        else:
            # Gap between two non-wake periods
            timeStart = findPosFromTimestamps(signalTime, p[0])
            timeEnd = findPosFromTimestamps(signalTime, p[1])

            numberOfPoints = timeEnd - timeStart + 1

            TimeSignal = np.ones([numberOfPoints]) * np.nan
            TimePoints = signalTime[timeStart:(timeEnd + 1)]

            InterLinesSignal.append(TimeSignal[1:-1])
            InterLinesPoints.append(TimePoints[1:-1])

            points.append(TimePoints[1:-1])
            baselineAprox.append(TimeSignal[1:-1])

    # STEP 6: PREPARE DATA FOR POLYNOMIAL FITTING

    # Concatenate all time points and baseline values
    points = np.concatenate(points)
    baselineAprox = np.concatenate(baselineAprox)

    # Combine into array and sort by time
    baselineAndPoints = np.transpose(np.array([baselineAprox, points]))
    baselineAndPointsSorted = baselineAndPoints[baselineAndPoints[:, 1].argsort()]

    baselineFinal = baselineAndPointsSorted[:, 0]
    times = baselineAndPointsSorted[:, 1]

    # STEP 7: FIT POLYNOMIAL TO NON-NAN BASELINE VALUES

    # Standardize time points for numerical stability
    valid_mask = np.invert(np.isnan(baselineFinal))
    mean = np.mean(times[valid_mask])
    std = np.std(times[valid_mask])

    # Normalize time points
    x = (times[valid_mask] - mean) / std

    # Fit 2nd order polynomial to valid baseline points
    cof = np.polyfit(x, baselineFinal[valid_mask], 2)

    # Evaluate polynomial for all time points
    fitbas = np.polyval(cof, (signalTime - mean) / std)

    return fitbas
