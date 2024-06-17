import numpy as np
from scipy import signal
import re
from Schema.UsefulFunctions import findPosFromTimestamps

def baseline(dff,timestamps, window,b):

    """
     This function is computing the baseline of a calcium singal using the
     5th percentile method of the 1Hz lowpass E: and then using the b file
     it fits a line only using the bouts of NREM and REM but excluding the
     wake bouts

     Inputs:
     dff: : input calcium dff
     timestamps: : timestamps of the acquisiition
     window: window around every point that is used to compute the percentile
     fs: sampling frequency

     Outputs:
     baseline: the computed baseline
     fitbas : the fitting line using only the NREM and REM bouts of the singal


    filter the singal at 1Hz to remove calcium activity

    design the low-pass filter
    """

    signaldff = dff
    signalTime = timestamps

    fs = int(np.round(1 / np.mean(np.diff(timestamps))))

    lpfilt = signal.firwin(1000, 0.01 / (fs / 2), pass_zero='lowpass')

    filtsing = signal.filtfilt(lpfilt, -1, signaldff)

    baseline = np.zeros_like(filtsing)

    print('Baseline Calculation started!')

    for i_timepoint in range(len(filtsing )):

        windowSignal = filtsing[np.max([0, i_timepoint - window * fs]):np.min([len(filtsing), (i_timepoint + window * fs)])]
        baseline[i_timepoint] = np.percentile(windowSignal, 5)

       # if np.mod(i_timepoint,1000) == 0:

        #    print('Calculation at:' + str(i_timepoint))

    print('Baseline Calculated!')

    TimesOfb = np.arange(0,(len(b))*4,4)

    exp = '[n2]+[^w]*'
    NoWake = re.finditer(exp, b)

    NoWakeStart = []
    NoWakeStop = []

    for nowake in NoWake:
        span = nowake.span()
        NoWakeStart .append(span[0])
        NoWakeStop.append(span[1])

    points = []
    baselineAprox = []

    interpIntervals = 0

    NREMBaselineSignal = []
    NREMBaselinePoints = []
    NREMBaselineTimesOfB = []

    NREMBaselineInterTimeStartPoint = []
    NREMBaselineInterTimeStartPointBfile = []
    NREMBaselineInterTimeEndPoint = []
    NREMBaselineInterTimeEndPointBfile = []

    for i, s in enumerate(zip(NoWakeStart,NoWakeStop)):

        StartIndexToKeep = TimesOfb[s[0]]
        StopIndexToKeep = TimesOfb[s[1]-1] + 4

        StartIndexDff = findPosFromTimestamps(signalTime,StartIndexToKeep)
        StopIndexDff = findPosFromTimestamps(signalTime,StopIndexToKeep)

        NREMBaselineSignal.append(baseline[StartIndexDff:StopIndexDff])
        NREMBaselinePoints.append(signalTime[StartIndexDff:StopIndexDff])
        NREMBaselineTimesOfB.append(np.arange(TimesOfb[s[0]],(TimesOfb[s[1]-1] + 4),4))

        points.append(signalTime[StartIndexDff:StopIndexDff])
        baselineAprox.append(baseline[StartIndexDff:StopIndexDff])

        if NoWakeStart[i] == 0:
            continue

        if i == 0 and not(NoWakeStart[i] == 0):

            interpIntervals += 1
            NREMBaselineInterTimeStartPoint.append(np.nan)
            NREMBaselineInterTimeStartPointBfile.append(np.nan)
            EndPoint = findPosFromTimestamps(signalTime,StartIndexToKeep)

            NREMBaselineInterTimeEndPoint.append(signalTime[EndPoint])
            NREMBaselineInterTimeEndPointBfile.append(TimesOfb[s[0]])
            continue

        interpIntervals += 1
        StartPoint = findPosFromTimestamps(signalTime, (TimesOfb[NoWakeStop[i-1]]))

        NREMBaselineInterTimeStartPoint.append(signalTime[StartPoint-1])
        NREMBaselineInterTimeStartPointBfile.append(TimesOfb[NoWakeStop[i-1]])

        Endoint = findPosFromTimestamps(signalTime, (TimesOfb[NoWakeStart[i]]))

        NREMBaselineInterTimeEndPoint.append(signalTime[Endoint])
        NREMBaselineInterTimeEndPointBfile.append(TimesOfb[NoWakeStart[i]])

        if i == (len(NoWakeStart) - 1) and not(NoWakeStop[i] == len(b)):

            interpIntervals += 1
            StartPoint =  findPosFromTimestamps(signalTime, (TimesOfb[NoWakeStop[i]]))

            NREMBaselineInterTimeStartPoint.append(signalTime[StartPoint-1])
            NREMBaselineInterTimeStartPointBfile.append(TimesOfb[NoWakeStop[i]])

            NREMBaselineInterTimeEndPoint.append(np.nan)
            NREMBaselineInterTimeEndPointBfile.append(np.nan)

    InterLinesSignal = []
    InterLinesPoints = []

    for i, p in enumerate(zip(NREMBaselineInterTimeStartPoint,NREMBaselineInterTimeEndPoint)):

        if np.isnan(p[0]):

            time = p[1]

            Pos = findPosFromTimestamps(signalTime, time)

            InterLinesSignal.append(np.ones([Pos])*np.nan)
            InterLinesPoints.append(signalTime[0:Pos])

            points.append(signalTime[0:Pos])
            baselineAprox.append(np.ones([Pos])*np.nan)

        elif np.isnan(p[1]):

            time = p[0]

            Pos = findPosFromTimestamps(signalTime, time)

            InterLinesSignal.append(np.ones([max(0,len(baseline)-Pos -1)])*np.nan)
            InterLinesPoints.append(signalTime[(Pos+1)::])

            points.append(signalTime[(Pos+1)::])
            baselineAprox.append(np.ones([max(0,len(baseline)-Pos -1)])*np.nan)
        else:

            timeStart = findPosFromTimestamps(signalTime, p[0])
            timeEnd = findPosFromTimestamps(signalTime, p[1])

            numberOfPoints = timeEnd - timeStart + 1

            TimeSignal = np.ones([numberOfPoints])*np.nan
            TimePoints = signalTime[timeStart:(timeEnd +1)]

            InterLinesSignal.append(TimeSignal[1:-1])
            InterLinesPoints.append(TimePoints[1:-1])

            points.append(TimePoints[1:-1])
            baselineAprox.append(TimeSignal[1:-1])

    points = np.concatenate(points)
    baselineAprox = np.concatenate(baselineAprox)

    baselineAndPoints = np.transpose(np.array([baselineAprox,points]))

    baselineAndPointsSorted = baselineAndPoints[baselineAndPoints[:,1].argsort()]

    baselineFinal = baselineAndPointsSorted[:,0]
    times = baselineAndPointsSorted[:,1]

    mean =  np.mean(times[np.invert(np.isnan(baselineFinal))])
    std = np.std(times[np.invert(np.isnan(baselineFinal))])

    x = (times[np.invert(np.isnan(baselineFinal))] - mean)/std

    cof = np.polyfit(x, baselineFinal[np.invert(np.isnan(baselineFinal))] , 2)

    fitbas = np.polyval(cof,(signalTime-mean)/std)

    return  fitbas


