# ============================================================
#  LC Projection Neurons In Sleep Data Injection Pipeline
#  ------------------------------------------------------------
#  Autopopulation functions related to the lcProj schema inlcuding
#  basic and advanved analysis of sleep electrophysiological data and
#  fiber photometry data
#
#  For more information  on DataJoint schemas please look at https://github.com/datajoint
#  Author: Georgios Foustoukos
#  Date: 14.08.2024
# ============================================================

# packages needed for this analysis
import datajoint as dj
import os, sys
import csv
import pathlib
import numpy as np
from tqdm import tqdm
from tqdm import trange
from natsort import natsorted
import pandas as pd
from pathlib import Path
import scipy.io as sio
from Schema import lcProj
import datetime
import numpy as np
import math
import glob
import mat73
import datetime
import re
from textwrap import wrap
from scipy.stats import norm
from Schema.UsefulFunctions import ElimMA, deltaDynamics, maDynamics, CorrectMirrorAnimals, findPosFromTimestamps,binning, MGT, BPM, EqualTimeSpentInNrem, b_num
from Schema.PeakDetection import PeakDetection
from Schema.baseline import baseline
from scipy.stats import zscore

import math
from scipy import signal
import regex as reg



def Hypnogram_autopopulation(self, key):
    """
    Hypnogram_autopopulation
    -------------
    Populates hypnogram sleep-state data into the database.

    Workflow:
    ---------
    1. Verify session belongs to LC or REMSD projects.
    2. Load scored b-file or approximate states from Hypnos output.
    3. Map state codes to numerical values for plotting (Wake=3, NREM=2, REM=1).
    4. Insert both raw and plot-ready state sequences into the database.

    Notes
    -----
    - Uses mat73 to read MATLAB files.
    - Supports both scored and automatically detected Hypnos states.

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    # function to read the b-file from file and inject it to DJ, it will automatically detect if the file is scored and
    # if not but recorded by Hypnos it will take the Hypnos states as an approximation of the b.

        if ('LC' in (lcProj.Session() & key).fetch('project_id')[0]) or ('REMSD' in (lcProj.Session() & key).fetch('project_id')[0]):

            print('Inserting the hypnogram of mouse: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

            fileToRead = (lcProj.Session & key).fetch1('file_name')

            if  not(fileToRead == 'No Photometry Data'):

                fileStates = mat73.loadmat(fileToRead, only_include='states')

                # if a file is scored then you read the b vector

                if 'bt.mat' in fileToRead:

                    print('--This file is scored')

                    file = mat73.loadmat(fileToRead,only_include = 'b')

                    key['bfile'] = file['b']

                    bfile = file['b']

                    hypToPlot = np.empty(len(bfile))

                    for i, v in enumerate(bfile):

                        if v == 'w' or v == '1' :
                            hypToPlot[i] = 3

                        elif v == 'n' or v == '2':
                            hypToPlot[i] = 2

                        elif v == 'r' or v == '3':

                            hypToPlot[i] = 1

                    key['bfile_to_plot'] = hypToPlot

                    #check if automatic detected states from Hypnos are there and inject them
                    if fileStates:

                        print('--This file was recorded by Hypnos')

                        key['hypnos_states'] = fileStates['states'][0,:]

                        states = fileStates['states'][0,:]

                        statesToPlot = np.empty(len(states))

                        for i,v  in enumerate(states):

                            if v == 1.0:
                                statesToPlot[i] = 3

                            elif v == 2.0:
                                statesToPlot[i] = 2

                            elif v == 3.0:
                                 statesToPlot[i] = 1

                            elif v == 0.0:
                                    statesToPlot[i] = np.nan

                        key['hypnos_states_to_plot'] = statesToPlot

                # if the file is not scored you use the Hypnos automatic detected states as the best appoximation
                elif fileStates:

                    print('--This file was recorded by Hypnos')

                    key['hypnos_states'] = fileStates['states'][0,:]

                    states = fileStates['states'][0,:]

                    statesToPlot = np.empty(len(states))

                    for i,v in enumerate(states):

                        if v == 1.0:
                            statesToPlot[i] = 3

                        elif v == 2.0:
                            statesToPlot[i] = 2

                        elif v == 3.0:
                             statesToPlot[i] = 1

                        elif v == 0.0:
                                statesToPlot[i] = np.nan

                    key['hypnos_states_to_plot'] = statesToPlot

                self.insert1(key)

def FibDFF_autopopulation(self, key):
    """
    FibDFF_autopopulation
    -------------
    Loads fiber photometry ΔF/F data from MATLAB files.

    Workflow:
    ---------
    1. Fetch session file path from the database.
    2. Load `dff` data matrix from .mat file.
    3. Extract timestamps and ΔF/F samples depending on file format.
    4. Insert processed ΔF/F and timestamps into the database.

    Notes
    -----
    - Handles both modern and legacy recording formats.

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    # function to read the dff singal already located in the .mat file after the signal has been demodulated and the DFF was computed

    print('Inserting the DFF of mouse: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

    fileToRead = (lcProj.Session & key).fetch1('file_name')

    if not(fileToRead == 'No Photometry Data'):

        fileDff = mat73.loadmat(fileToRead, only_include='dff')

        dff = fileDff['dff']
        if  dff.shape[0] == 2:

            dffSamples =  dff[1,:]
            timestamps = dff[0,:]

        elif dff.shape[0] == 432000:

            print('Session recorded with the old system')
            dffSamples = dff
            timestamps = np.arange(0,12*3600,1/10)


        key['dff'] =  dffSamples
        key['timestamps'] = timestamps

        self.insert1(key)

def REMLatency_autopopulation(self, key):
    """
    REMLatency_autopopulation
    -------------
    Computes the REM latency from a scored file using the hypnogram (w, r, n as three sleep states)

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. Find the first time that consolidated NREM sleep is present in the string (using finditer)
    3. Find the first time that consolidated REM sleep is present in the string (using finditer)
    4. Compute the difference in time between the previous 2 as sleep rem_latency
    5. Recompute the same number after long waking episoded have been removed for comparison
    6. Insert data into database

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    # This a function which computes the latency to consolidated REM after SD or SSD.

    if (('SD' in (lcProj.Session() & key).fetch('project_id')[0]) or ('Baseline' in (lcProj.Session() & key).fetch('project_id')[0])):

        print('Calculating REM latency: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

        NumOfScoredWind = 3

        bfile = (lcProj.Hypnogram & key).fetch1('bfile')

        # Detect the first "n" (or 1) in the b file
        TransToNREM = re.finditer('[w1][n2]{1,}[w1n2r3]',bfile)

        TransSpans = []

        for idx, trans in enumerate(TransToNREM):

                Beg = list(trans.span())[0]+1
                End = list(trans.span())[1]-1
                TransSpans.append([Beg,End])

        TransTime = TransSpans[0][0]

        # find the first consolidated REM as 3 scoring windows of "r" (or 3)
        ConsolidatedREM = re.finditer('[n2][r3]{' + str(NumOfScoredWind) + ',}[w1r3]',bfile)

        REMSpans = []

        for idx, rem in enumerate(ConsolidatedREM):

                Beg = list(rem.span())[0]+1
                End = list(rem.span())[1]-1
                REMSpans.append([Beg,End])

        REMlatency = (REMSpans[0][0] - TransTime) *4

        # correct REM latency for long wake episodes

        bfileBetween = bfile[TransTime:(REMSpans[0][0]+1)]

        WakeMiddle = re.finditer('[n2][w1]{4,}[n2]',bfileBetween)

        WakeEnd = re.finditer('[n2][w1]{4,}' + '$',bfileBetween)

        WakeBeg= re.finditer('^' + '[w1]{4,}[n2]',bfileBetween)

        WakeAfterREM = re.finditer('[r3][w1]{4,}[n2]',bfileBetween)

        WakeAfterREMEnd = re.finditer('[r3][w1]{4,}' + '$',bfileBetween)

        WakeSpans = []

        for idx, w in enumerate(WakeMiddle):

            Beg = list(w.span())[0]+1
            End = list(w.span())[1]-1
            WakeSpans.append([Beg,End])

        for idx, w in enumerate(WakeBeg):

            Beg = list(w.span())[0]
            End = list(w.span())[1]-1
            WakeSpans.append([Beg,End])

        for idx, w in enumerate(WakeEnd):

            Beg = list(w.span())[0]+1
            End = list(w.span())[1]
            WakeSpans.append([Beg,End])

        for idx, w in enumerate(WakeAfterREM):

            Beg = list(w.span())[0]+1
            End = list(w.span())[1]-1
            WakeSpans.append([Beg,End])

        for idx, w in enumerate(WakeAfterREMEnd):

            Beg = list(w.span())[0]+1
            End = list(w.span())[1]
            WakeSpans.append([Beg,End])

        WakeSpansArray =  np.array(WakeSpans)

        bfileBetweenWakeCor = bfileBetween

        for i, w in enumerate(WakeSpansArray):

            Temp =  bfileBetweenWakeCor[0:w[0]] + '0' * (w[1]-w[0])  + bfileBetweenWakeCor [w[1]:]

            bfileBetweenWakeCor  = Temp


        Ns=  re.findall('n',bfileBetweenWakeCor)
        Twos  =  re.findall('2',bfileBetweenWakeCor)
        Ws = re.findall('w',bfileBetweenWakeCor)
        Ones = re.findall('1',bfileBetweenWakeCor)
        Rs = re.findall('r',bfileBetweenWakeCor)
        Threes = re.findall('3',bfileBetweenWakeCor)

        REMLatencyWakeCor = (len(Ns) + len(Twos) + len(Ws) + len(Ones) + len(Rs) + len(Threes) -1)*4

        key['consolidated_rem'] = NumOfScoredWind*4

        key['rem_latency'] = REMlatency

        key['rem_latency_wake_cor'] = REMLatencyWakeCor


        if lcProj.OptoSession & key:

            sessionType = (lcProj.OptoSession & key).fetch1('session_type')

        elif lcProj.SDSession & key:

            sessionType = (lcProj.SDSession & key).fetch1('session_type')

        elif lcProj.SSDSession & key:

            sessionType = (lcProj.SSDSession & key).fetch1('session_type')

        elif lcProj.REMSDSession & key:

            sessionType = (lcProj.REMSDSession & key).fetch1('session_type')

        elif lcProj.BaselineSession & key:

            sessionType = (lcProj.BaselineSession & key).fetch1('session_type')

        key['session_info'] = sessionType

        self.insert1(key)

def DeltaDynamics_autopopulation(self, key):
    """
    DeltaDynamics_autopopulation
    -------------
    Computes the delta dynamics using the bipolarised EEG signals after the mouse transition to sleep

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. Find the time of transition to sleep (consolidated NREM) in case of SD/SSD or start from O in case of a baseline recording
    3. Pass this to the deltaDynamics function for computation
    4. Insert data into database

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    if ('SD' in (lcProj.Session() & key).fetch('project_id')[0]) and (not ('REMSD' in (lcProj.Session() & key).fetch('project_id')[0])):

        print('Calculating delta dynamics for ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

        bfile = (lcProj.Hypnogram & key).fetch1('bfile')

        fileToRead = (lcProj.Session & key).fetch1('file_name')

        matFile = mat73.loadmat(fileToRead)

        # find the first "n" or "2" in the file
        TransToNREM = re.finditer('[w1][n2]{1,}[w1n2r3]',bfile)

        TransSpans = []

        # read the traces in order to calculate the bipolarised EEG

        for idx, trans in enumerate(TransToNREM):

                Beg = list(trans.span())[0]+1
                End = list(trans.span())[1]-1
                TransSpans.append([Beg,End])

        TransTime = TransSpans[0][0]

        traces = matFile['traces']

        EEGP = traces[0,:]
        EEGF = traces[1,:]


        EEGBip = EEGP - EEGF

        # this deltaDynamics function calculated the dynamics of delta

        deltaPowerPointsWakeArrayNorm, deltaPowerPointsNREMArrayNorm, deltaPowerPointsREMArrayNorm, timePositionArray = deltaDynamics (bfile,TransTime,EEGBip, 8, 1000)

        key['delta_power_nrem'] = deltaPowerPointsNREMArrayNorm
        key['delta_power_rem'] = deltaPowerPointsREMArrayNorm
        key['delta_power_wake'] = deltaPowerPointsWakeArrayNorm
        key['timepoints'] = timePositionArray

        self.insert1(key)

        # same calculation as before but for the fiber photometry data

    elif ('Baseline' in (lcProj.Session() & key).fetch('project_id')[0]):

        print('Calculating delta dynamics for ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

        bfile = (lcProj.Hypnogram & key).fetch1('bfile')

        fileToRead = (lcProj.Session & key).fetch1('file_name')

        matFile = mat73.loadmat(fileToRead)

        traces = matFile['traces']

        EEGP = traces[0,:]
        EEGF = traces[1,:]

        EEGBip = EEGP - EEGF

        deltaPowerPointsWakeArray, deltaPowerPointsNREMArray, deltaPowerPointsREMArray, timePositionArray = deltaDynamics (bfile,0,EEGBip, 12, 1000)

        key['delta_power_nrem'] = deltaPowerPointsNREMArray
        key['delta_power_rem'] = deltaPowerPointsREMArray
        key['delta_power_wake'] = deltaPowerPointsWakeArray
        key['timepoints'] = timePositionArray

        self.insert1(key)


def MADynamics_autopopulation(self, key):
    """
    MADynamics_autopopulation
    -------------
    Computes the dynamics of microarousals in SD/SDD and baseline recordings

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. For the SD or SSD files find the time the that the first consolidated NREM is present
    3. Find the MAs using the ElimMA function
    4. Use the maDynamics function to compute the density
    4. Insert data into database

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    if ('SD' in (lcProj.Session() & key).fetch('project_id')[0]) and (not ('REMSD' in (lcProj.Session() & key).fetch('project_id')[0])):

        print('Calculating MA dynamics for ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

        bfile = (lcProj.Hypnogram & key).fetch1('bfile')

        bfileMACor, MASpansArray, PerMAs = ElimMA(bfile,12)

        fileToRead = (lcProj.Session & key).fetch1('file_name')

        matFile = mat73.loadmat(fileToRead)

        # find the first "n" or "2" in the file
        TransToNREM = re.finditer('[w1][n2]{1,}[w1n2r3]',bfile)

        TransSpans = []

        for idx, trans in enumerate(TransToNREM):

                Beg = list(trans.span())[0]+1
                End = list(trans.span())[1]-1
                TransSpans.append([Beg,End])

        TransTime = TransSpans[0][0]

        # this maDynamics function calculates the dynamics of the MA's

        MADensity, timePositionArray = maDynamics(bfile,TransTime,8)

        key['ma_density'] = MADensity
        key['timepoints'] = timePositionArray
        key['per_ma'] = PerMAs

        self.insert1(key)


    elif ('Baseline' in (lcProj.Session() & key).fetch('project_id')[0]):

            print('Calculating MA dynamics for ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

            bfile = (lcProj.Hypnogram & key).fetch1('bfile')

            bfileMACor, MASpansArray, PerMAs = ElimMA(bfile,12)

            MADensity, timePositionArray = maDynamics(bfile,0,12)

            key['ma_density'] = MADensity
            key['timepoints'] = timePositionArray
            key['per_ma'] = PerMAs

            self.insert1(key)


    elif  ('REMSD' in (lcProj.Session() & key).fetch('project_id')[0]):


        if  ((lcProj.REMSDSession & key).fetch1('session_type') == 'REM sleep deprivation yoked control'):

            print('Calculating MA dynamics for ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date') + ', correction of the mirror control animal')

            PairsDict = {'RD3':'RD4',
                         'RD4':'RD3',
                         'RD6':'RD7',
                         'RD7':'RD6',
                         'RD8':'RD9',
                         'RD9':'RD8',
                         'RD14':'RD15',
                         'RD15':'RD14',
                         'RD16':'RD17',
                         'RD17':'RD16'}


            mouseMirror =  key['mouse_name']

            keySD = {'mouse_name': PairsDict[mouseMirror]}

            HypREMSDPair = (lcProj.Hypnogram & (lcProj.REMSDSession & keySD & 'session_type = "REM sleep deprivation"')).fetch1('bfile')

            HypMirrorPair =  (lcProj.Hypnogram & key).fetch1('bfile')

            HypMirrorPairCorrected =  CorrectMirrorAnimals (HypREMSDPair,HypMirrorPair, 6)

            bfileMACor, MASpansArray, PerMAs = ElimMA(HypMirrorPairCorrected,12)

            MADensity, timePositionArray = maDynamics(HypMirrorPairCorrected ,0,12)

            key['ma_density'] = MADensity
            key['timepoints'] = timePositionArray
            key['per_ma'] = PerMAs

            self.insert1(key)


        else:

            print('Calculating MA dynamics for ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

            bfile = (lcProj.Hypnogram & key).fetch1('bfile')

            bfileMACor, MASpansArray, PerMAs = ElimMA(bfile,12)

            MADensity, timePositionArray = maDynamics(bfile,0,12)

            key['ma_density'] = MADensity
            key['timepoints'] = timePositionArray
            key['per_ma'] = PerMAs

            self.insert1(key)


def TimeSpentTransitionsDynamics_autopopulation(self, key):
    """
    TimeSpentTransitionsDynamics_autopopulation
    -------------
    Computes the time spent in each sleep state and transitions dynamics using the hypnogram scored file

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. Extracts the bouts of NREM, REM and Wake using character detection in the hypnogram files using the full file or a windowed version of it
    3. Computes the time spent in each sleep state usign the extracted bouts
    4. Compute the number of transitions to REM sleep for the windowed signal also
    5. Insert results into the database (the bouts are also inserted)

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    print('Doing: ' + key['mouse_name'] + '---')
    print('Doing: ' + key['session_date'])

    bfile = (lcProj.Hypnogram & key).fetch1('bfile')

    # find the amount of different letters for n,r,w, 1,2,3 for their artifacts
    RemBouts1 = re.findall('r',bfile )
    RemBouts2 = re.findall('3',bfile )

    NoRemBouts1 = re.findall('n',bfile)
    NoRemBouts2 = re.findall('2',bfile)

    AwakeBouts1 = re.findall('w',bfile)
    AwakeBouts2 = re.findall('1',bfile)

    RemBoutsTotal = len(RemBouts1) +  len(RemBouts2)

    NoRemBoutsTotal =  len(NoRemBouts1) + len(NoRemBouts2)

    AwakeBoutsTotal =  len(AwakeBouts1) +  len(AwakeBouts2)

    timeInRem = RemBoutsTotal/len(bfile)
    timeInNoRem = NoRemBoutsTotal/len(bfile)
    timeInAwake = AwakeBoutsTotal/len(bfile)

    key['time_spent_norem'] = timeInNoRem
    key['time_spent_rem'] = timeInRem
    key['time_spent_awake'] = timeInAwake

    window = key['window']

    print('Calculating: ' + key['mouse_name'] + ' for session: ' +
    (lcProj.Session & key).fetch1('session_date') +' and window at: ' + str(window)  + 's')

    RealTime = len(bfile)*4

    #find the bouts of r,n and w (it looks complicated but it makes sure you get the bouts at the beginning and the ends of the file)
    rem = re.findall('[r*3*]*r+[3*r*]*|[r*3*]*3+[3*r*]*',bfile)
    RemEpochs = []

    for r in rem :
        RemEpochs.append(len(r)*4)

    RemEpochsArray = np.array(RemEpochs)

    norem =re.findall('[n*2*]*n+[2*n*]*|[n*2*]*2+[2*n*]*',bfile)
    NoRemEpochs = []
    for n in norem :
        NoRemEpochs.append(len(n)*4)

    NoRemEpochsArray = np.array(NoRemEpochs)

    awake = re.findall('[w*1*]*w+[1*w*]*|[w*1*]*1+[1*w*]*',bfile)
    AwakeEpochs = []
    for w in awake:
        AwakeEpochs.append(len(w)*4)

    AwakeEpochsArray = np.array(AwakeEpochs)

    key['noremepochs'] = NoRemEpochsArray
    key['remepochs'] = RemEpochsArray
    key['awakeepochs'] = AwakeEpochsArray


    # This analysis is in case we would like to window the stimulation window to get the dynamics

    WindowLengthInCharacters = int(window/4)

    #window in non-overlapping windows
    statesWindowed = wrap(bfile,WindowLengthInCharacters)

    RemWindowed = np.empty(len(statesWindowed))
    NoRemWindowed = np.empty(len(statesWindowed))
    AwakeWindowed = np.empty(len(statesWindowed))
    TransRemWindowed = np.empty(len(statesWindowed))

    for i,c in enumerate(statesWindowed):

        RemTransitions = re.findall('[n2w1][r3]', c)

        NREMBouts1 = re.findall('n', c)
        NREMBouts2 = re.findall('2', c)

        REMTransitionsTotal = len(RemTransitions)

        NREMTotalTime = (len(NREMBouts1) + len(NREMBouts2)) * 4 / 60

        if NREMTotalTime == 0:
            transitionsREMWin = np.nan
        else:
            transitionsREMWin = REMTransitionsTotal / NREMTotalTime

        TransRemWindowed[i] = transitionsREMWin

        RemInChunk1 = re.findall('r+',c)
        RemInChunk2 = re.findall('3+',c)
        RemInChunk = RemInChunk1 + RemInChunk2

        RemEpochsInChunk = []
        for r in RemInChunk :
            RemEpochsInChunk.append(len(r)*4)

        RemEpochsInChunkArray = np.array(RemEpochsInChunk)

        NoRemInChunk1 = re.findall('n+',c)
        NoRemInChunk2 = re.findall('2+',c)
        NoRemInChunk = NoRemInChunk1 + NoRemInChunk2

        NoRemEpochsInChunk = []
        for n in NoRemInChunk :
            NoRemEpochsInChunk.append(len(n)*4)

        NoRemEpochsInChunkArray = np.array(NoRemEpochsInChunk)

        AwakeInChunk1 = re.findall('w+',c)
        AwakeInChunk2 = re.findall('1+',c)
        AwakeInChunk = AwakeInChunk1 + AwakeInChunk2
        AwakeEpochsInChunk = []

        for w in AwakeInChunk :
            AwakeEpochsInChunk.append(len(w)*4)

        AwakeEpochsInChunkArray = np.array(AwakeEpochsInChunk)

        RemWindowed[i] = np.sum(RemEpochsInChunkArray)/window*100
        NoRemWindowed[i] = np.sum(NoRemEpochsInChunkArray)/window*100
        AwakeWindowed[i] = np.sum(AwakeEpochsInChunkArray)/window*100


    key['time_spent_rem_windowed'] = RemWindowed
    key['time_spent_norem_windowed'] = NoRemWindowed
    key['time_spent_awake_windowed'] = AwakeWindowed
    key['rem_transitions_windowed'] =  TransRemWindowed

    self.insert1(key)


def  DFFSleepStates_autopopulation(self, key):
    """
    DFFSleepStates_autopopulation
    -------------
    Compute the mean DF/F0 value of the LC fiber photometry signal for each sleep state

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. Reduce the DF/F0 only to certain amount of hours (using the window value) to avoid photo-bleaching issues and Z-scored it
    3. Use the hypnogram to find the timepoints of each sleep state and get the corresponding DF/F0 values
    4. Safely concatenate and mean the data
    5. Insert results into the database.

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    print('Computing mouse: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

    print('Computing the mean DFF for every sleep state during the photometry recordings for the first',str(key['window']),'hours')

    bfile = (lcProj.Hypnogram & key).fetch1('bfile')
    dff = (lcProj.FibDFF & key).fetch1('dff')
    timestamps = (lcProj.FibDFF & key).fetch1('timestamps')

    window = key['window']

    lengthCheck = (timestamps[-1] - timestamps[0])/3600

    print(lengthCheck)

    if lengthCheck > window:

        posToKeepDff = findPosFromTimestamps(timestamps, window*3600)

        dffKeep = dff[0:posToKeepDff]
        bKeep =  bfile[0:int((window*3600/4))]

        dffKeepZScored = ((dffKeep  - np.mean(dffKeep))/np.std(dffKeep))

        REMbouts = re.finditer('[r3]{3,}',bKeep)
        REMSpans = []
        DffREM  = []
        DffREMZScored  = []

        for idx, trans in enumerate(REMbouts):

            Beg = list(trans.span())[0]
            End = list(trans.span())[1]

            BegDFF = findPosFromTimestamps(timestamps, Beg*4)
            EndDFF = findPosFromTimestamps(timestamps, End*4)

            REMSpans.append([BegDFF,EndDFF])
            #DffREM.append(np.mean(dffKeep[BegDFF:EndDFF]))
            DffREM.append(dffKeep[BegDFF:EndDFF])
            #DffREMZScored.append(np.mean(dffKeepZScored[BegDFF:EndDFF]))
            DffREMZScored.append(dffKeepZScored[BegDFF:EndDFF])

        NREMbouts = re.finditer('[n2]{3,}',bKeep)
        NREMSpans = []
        DffNREM = []
        DffNREMZScored  = []

        for idx, trans in enumerate(NREMbouts):
            Beg = list(trans.span())[0]
            End = list(trans.span())[1]

            BegDFF = findPosFromTimestamps(timestamps, Beg * 4)
            EndDFF = findPosFromTimestamps(timestamps, End * 4)

            NREMSpans.append([BegDFF,EndDFF])
            #DffNREM.append(np.mean(dffKeep[BegDFF:EndDFF]))
            DffNREM.append(dffKeep[BegDFF:EndDFF])
            #DffNREMZScored.append(np.mean(dffKeepZScored[BegDFF:EndDFF]))
            DffNREMZScored.append(dffKeepZScored[BegDFF:EndDFF])

        Wakebouts = re.finditer('[w1]{4,}',bKeep)
        WakeSpans = []
        DffWake = []
        DffWakeZScored  = []

        for idx, trans in enumerate(Wakebouts):
            Beg = list(trans.span())[0]
            End = list(trans.span())[1]

            BegDFF = findPosFromTimestamps(timestamps, Beg * 4)
            EndDFF = findPosFromTimestamps(timestamps, End * 4)

            WakeSpans.append([BegDFF,EndDFF])
            #DffWake.append(np.mean(dffKeep[BegDFF:EndDFF]))
            DffWake.append(dffKeep[BegDFF:EndDFF])
            #DffWakeZScored.append(np.mean(dffKeepZScored[BegDFF:EndDFF]))
            DffWakeZScored.append(dffKeepZScored[BegDFF:EndDFF])

        # key['mean_rem_zscored'] = np.mean(DffREMZScored)
        # key['mean_nrem_zscored'] = np.mean(DffNREMZScored)
        # key['mean_wake_zscored'] = np.mean(DffWakeZScored)
        # key['mean_rem'] = np.mean(DffREM)
        # key['mean_nrem'] = np.mean(DffNREM)
        # key['mean_wake'] = np.mean(DffWake)

        def safe_mean_concat(data_list):
            return np.mean(np.concatenate(data_list)) if data_list else np.nan

        key['mean_rem_zscored'] = safe_mean_concat(DffREMZScored)
        key['mean_nrem_zscored'] = safe_mean_concat(DffNREMZScored)
        key['mean_wake_zscored'] = safe_mean_concat(DffWakeZScored)
        key['mean_rem'] = safe_mean_concat(DffREM)
        key['mean_nrem'] = safe_mean_concat(DffNREM)
        key['mean_wake'] = safe_mean_concat(DffWake)

        self.insert1(key)


def NremRemMeanDFF_autopopulation(self, key):
    """
    NremRemMeanDFF_autopopulation
    -------------
    Computes the mean DF/F0 values either during a REM episode or between two REM Episodes (in the NREM sleep between the two REM sleeps)

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. Finds the REM bouts in the hypnogram and computes the DF/F0 values during those bouts
    3. Using the found REM bouts extracts the NREM sleep between two consecutive REM bouts and computes the mean DF/F0 in those
    4. Means the data and inserts in the database

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    print('Computing mouse: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

    bfile = (lcProj.Hypnogram & key).fetch1('bfile')
    dff = (lcProj.FibDFF & key).fetch1('dff')
    timestamps = (lcProj.FibDFF & key).fetch1('timestamps')

    REMbouts = re.finditer('[r3]{3,}', bfile)
    DffREMZScoredBinned  = []

    for idx, trans in enumerate(REMbouts):

        Beg = list(trans.span())[0]
        End = list(trans.span())[1]

        BegDFF = findPosFromTimestamps(timestamps, Beg*4)
        EndDFF = findPosFromTimestamps(timestamps, End*4)

        dffREM = dff[BegDFF:EndDFF]

        dffREMZscored = (( dffREM - np.mean( dffREM ))/np.std(dffREM))

        dffREMZscoredBinned = binning(dffREMZscored)

        DffREMZScoredBinned.append(dffREMZscoredBinned)

    print('REMs found: ' + str(np.array(DffREMZScoredBinned).shape[0]))
    key['during_rem'] = np.mean(np.array(DffREMZScoredBinned),0)

    InterREMbouts = reg.finditer('([r3][w1n2]+[r3])', bfile, overlapped=True)

    DffInterREMAll = []

    for idx, trans in enumerate(InterREMbouts):

        DffInterREM = []

        Beg = list(trans.span())[0] + 1
        End = list(trans.span())[1] - 1

        bInBetween = bfile[Beg:End]

        NremInBetween = re.finditer('[n2]{3,}', bInBetween)

        NremInBetweenFlag = len(re.findall('[n2]{3,}', bInBetween))

        if NremInBetweenFlag > 0:

            BegDFF =  Beg * 4

            for idx, trans in enumerate(NremInBetween):

                BegNREM = list(trans.span())[0]
                EndNREM = list(trans.span())[1]

                BegDFFNREM = findPosFromTimestamps(timestamps, BegDFF + (BegNREM * 4))
                EndDFFNREM = findPosFromTimestamps(timestamps, BegDFF + (EndNREM * 4))

                DffInterREM.append(dff[BegDFFNREM:EndDFFNREM])

            DffInterREM = np.concatenate(DffInterREM)

            DffInterREMZscored = ((DffInterREM - np.mean(DffInterREM)) / np.std(DffInterREM))

            DffInterREMZscoredBinned = binning(DffInterREMZscored)

            DffInterREMAll.append(DffInterREMZscoredBinned)

    DffInterREMAllBinned = np.array(DffInterREMAll)

    print('InterREMs found: ' + str(DffInterREMAllBinned.shape[0]))
    key['inter_rem'] = np.mean(DffInterREMAllBinned,0)

    InterREMbouts = reg.finditer('[r3]{3,}', bfile)

    DffInterREMAll = []
    Rlength = []
    PointStart = []
    PointEnd = []

    for interrembout in InterREMbouts:

        PointStart.append(int(interrembout.span()[0]))
        PointEnd.append( int(interrembout.span()[1]))

    for i in range(len(PointStart)-1):

        StartOne = PointEnd[i]
        EndOne = PointStart[i+1]

        DffInterREM = []

        bInBetween = bfile[StartOne:EndOne]

        CurrentRemLength = len(re.findall('[r3]',bfile[PointStart[i]:PointEnd[i]]))*4

        NremInBetween = re.finditer('[n2]{1,}', bInBetween)

        NremInBetweenFlag = len(re.findall('[n2]{1,}', bInBetween))

        if NremInBetweenFlag > 0:

            BegDFF =  StartOne  * 4

            Rlength.append(CurrentRemLength)

            for idx, trans in enumerate(NremInBetween):

                BegNREM = list(trans.span())[0]
                EndNREM = list(trans.span())[1]

                BegDFFNREM = findPosFromTimestamps(timestamps, BegDFF + (BegNREM * 4))
                EndDFFNREM = findPosFromTimestamps(timestamps, BegDFF + (EndNREM * 4))

                DffInterREM.append(dff[BegDFFNREM:EndDFFNREM])

            DffInterREM = np.concatenate(DffInterREM)

            DffInterREMZscored = ((DffInterREM - np.mean(DffInterREM)) / np.std(DffInterREM))

            DffInterREMZscoredBinned = binning(DffInterREMZscored)

            DffInterREMAll.append(DffInterREMZscoredBinned)

    DffInterREMAllBinned = np.array(DffInterREMAll)

    key['inter_rem_length'] = DffInterREMAllBinned
    key['rem_length'] = Rlength
    self.insert1(key)


def LCPeaks_autopopulation(self, key):
    """
    LCPeaks_autopopulation
    -------------
    Auto-populates processed neuroscience experiment data into a DataJoint table.

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. Perform task-specific preprocessing or calculations.
    3. Insert results into the database.

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    print('Computing mouse: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

    dff = (lcProj.FibDFF & key).fetch1('dff')
    b_file = (lcProj.Hypnogram & key).fetch1('bfile')
    time = (lcProj.FibDFF & key).fetch1('timestamps')
    dff = dff.copy()
    fitbas = baseline(dff,time,60,b_file)
    dffCor = dff - fitbas

    peaksLCFreq, peaks, flags = PeakDetection(dffCor, b_file, time)

    key['lc_peaks'] = peaks
    key['peaks_flag'] = flags

    self.insert1(key)


def MGT_autopopulation(self, key):
    """
    MGT_autopopulation
    -------------
    Auto-populates processed neuroscience experiment data into a DataJoint table.

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. Perform task-specific preprocessing or calculations.
    3. Insert results into the database.

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    print('Computing mouse: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

    fileName = (lcProj.Session & key).fetch1('file_name')

    print('Loading traces---')
    matFile = mat73.loadmat(fileName)

    traces = matFile['traces']

    EEGP = traces[0, :]
    EEGF = traces[1, :]

    EEGBip = EEGP - EEGF

    EEGBip = signal.resample(EEGBip, int(EEGBip.shape[0]/5))

    print('Computing delta---')
    WTDelta = MGT(EEGBip, 200, 1.5, 4, 0.25)
    WTDelta = np.abs(WTDelta)
    WTDelta = np.mean(WTDelta, 0)
    WTDelta = signal.resample(WTDelta, int(WTDelta.shape[0] / 2))

    print('Computing sigma---')
    WTSigma = MGT(EEGBip, 200, 10, 15, 0.25)
    WTSigma = np.abs(WTSigma)
    WTSigma = np.mean(WTSigma, 0)
    WTSigma = signal.resample(WTSigma, int(WTSigma.shape[0] / 2))

    print('Computing gamma---')
    WTGamma = MGT(EEGBip, 200, 60, 80, 0.25)
    WTGamma = np.abs(WTGamma)
    WTGamma = np.mean(WTGamma, 0)
    WTGamma = signal.resample(WTGamma, int(WTGamma.shape[0] / 2))

    key['delta'] = WTDelta
    key['sigma'] = WTSigma
    key['gamma'] = WTGamma

    self.insert1(key)


def HR_autopopulation(self, key):
    """
    HR_autopopulation
    -------------
    Auto-populates processed neuroscience experiment data into a DataJoint table.

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. Perform task-specific preprocessing or calculations.
    3. Insert results into the database.

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    print('Computing mouse: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

    fileName = (lcProj.Session & key).fetch1('file_name')

    print('Loading traces---')
    matFile = mat73.loadmat(fileName)

    traces = matFile['traces']

    EMG = traces[3,:] - traces[2,:]

    b = (lcProj.Hypnogram & key).fetch1('bfile')

    bpm = BPM(EMG,b,1000, True)

    key['heart_rate'] = bpm

    self.insert1(key)


def LCFreq_autopopulation(self, key):
    """
    LCFreq_autopopulation
    -------------
    Auto-populates processed neuroscience experiment data into a DataJoint table.

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. Perform task-specific preprocessing or calculations.
    3. Insert results into the database.

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    print('Computing mouse: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

    keyToCompute = (lcProj.Session & key).fetch()

    print(keyToCompute['project_id'])

    if ('SD' in  keyToCompute['project_id'][0]) or ('SSD' in keyToCompute['project_id'][0]):
        print('This is a deprivation session')

        dff = (lcProj.FibDFF & key).fetch1('dff')
        b_file = (lcProj.Hypnogram & key).fetch1('bfile')
        time = (lcProj.FibDFF & key).fetch1('timestamps')
        dff = dff.copy()
        fitbas = baseline(dff,time,60,b_file)
        dffCor = dff - fitbas

        peaksLCFreq, peaks, flags = PeakDetection(dffCor, b_file, time)

        # find the first "n" or "2" in the file
        TransToNREM = re.finditer('[w1][n2]{1,}[w1n2r3]',b_file)

        TransSpans = []

        for idx, trans in enumerate(TransToNREM):
            Beg = list(trans.span())[0] + 1
            End = list(trans.span())[1] - 1
            TransSpans.append([Beg, End])

        TransTime = TransSpans[0][0]

        peaksToKeep = []
        peaksToKeepPos = []

        for p in peaksLCFreq:

            if time[p] <=  TransTime:

                peaksToKeep.append(np.nan)
                peaksToKeepPos.append(np.nan)
            else:
                peaksToKeep.append(time[p])
                peaksToKeepPos.append(p)

        peaksToKeep = np.array(peaksToKeep)
        peaksToKeepPos = np.array(peaksToKeepPos)

        Freq = np.hstack([1./np.diff(peaksToKeep), np.nan])

        bNum = b_num(b_file, time)
        CheckPeaks = np.array(bNum).astype(np.float64)
        CheckPeaks[CheckPeaks != 2] = np.nan
        ToDelete = []
        for p in range(len(peaksToKeepPos) - 1):

            if not(np.isnan(peaksToKeepPos[p])):

                if np.sum(np.isnan(CheckPeaks[peaksToKeepPos[p]:(peaksToKeepPos[p + 1] + 1)])):
                    ToDelete.append(1)
                else:
                    ToDelete.append(0)
            else:
                ToDelete.append(1)

        ToDelete.append(0)

        FreqCor = np.delete(Freq, np.array(ToDelete) == 1)
        PeakTimesCor = np.delete(peaksToKeep, np.array(ToDelete) == 1)

        windowsBorders = EqualTimeSpentInNrem(TransTime, b_file, 8)

        FreqWindowed = []
        WindowsTimes = []

        for w in windowsBorders:

            start = w[0]*4
            end = w[1]*4

            WindowsTimes.append(np.mean([start, end]))

            FreqWindow = FreqCor[np.logical_and(PeakTimesCor>=start,PeakTimesCor<=end)]

            FreqMean = np.nanmean(FreqWindow)

            FreqWindowed.append(FreqMean)

        key['lc_freq'] = np.array(FreqWindowed)
        key['window_times'] = np.array( WindowsTimes)
        self.insert1(key)

    elif (keyToCompute['project_id'][0] == 'LC Photometry of Projection Neurons Baseline'):

        print('This is baseline session')
        dff = (lcProj.FibDFF & key).fetch1('dff')
        b_file = (lcProj.Hypnogram & key).fetch1('bfile')
        time = (lcProj.FibDFF & key).fetch1('timestamps')
        dff = dff.copy()
        fitbas = baseline(dff,time,60,b_file)
        dffCor = dff - fitbas

        peaksLCFreq, peaks, flags = PeakDetection(dffCor, b_file, time)

        peaksTimes = []

        for p in peaksLCFreq:

            peaksTimes.append(time[p])

        peaksTimes = np.array(peaksTimes)

        Freq = np.hstack([1. / np.diff(peaksTimes),np.nan])

        bNum = b_num(b_file, time)
        CheckPeaks = np.array(bNum).astype(np.float64)
        CheckPeaks[CheckPeaks != 2] = np.nan
        ToDelete = []
        for p in range(len(peaksLCFreq) - 1):

            if np.sum(np.isnan(CheckPeaks[peaksLCFreq[p]:(peaksLCFreq[p + 1] + 1)])):
                ToDelete.append(1)
            else:
                ToDelete.append(0)

        ToDelete.append(0)

        FreqCor = np.delete(Freq, np.array(ToDelete) == 1)
        PeakTimesCor = np.delete(peaksTimes, np.array(ToDelete) == 1)

        windowsBorders = EqualTimeSpentInNrem(0, b_file, 12)

        FreqWindowed = []
        WindowsTimes = []

        for w in windowsBorders:
            start = w[0] * 4
            end = w[1] * 4

            WindowsTimes.append(np.mean([start, end]))

            FreqWindow = FreqCor[np.logical_and(PeakTimesCor >= start,   PeakTimesCor <= end)]

            FreqMean = np.nanmean(FreqWindow)

            FreqWindowed.append(FreqMean)

        key['lc_freq'] = np.array(FreqWindowed)
        key['window_times'] = np.array(WindowsTimes)

        self.insert1(key)

    elif (keyToCompute['project_id'][0] == 'LC Photometry of Projection Neurons combined with Fear Conditioning'):

        if not('Before' in ((lcProj.FCSession & key).fetch1('session_type'))):

            print('This is FC session')
            if lcProj.FibDFF & key:

                dff = (lcProj.FibDFF & key).fetch1('dff')
                b_file = (lcProj.Hypnogram & key).fetch1('bfile')
                time = (lcProj.FibDFF & key).fetch1('timestamps')
                dff = dff.copy()
                fitbas = baseline(dff, time, 60, b_file)
                dffCor = dff - fitbas

                peaksLCFreq, peaks, flags = PeakDetection(dffCor, b_file, time)

                peaksTimes = []

                for p in peaksLCFreq:
                    peaksTimes.append(time[p])

                peaksTimes = np.array(peaksTimes)

                Freq = np.hstack([1. / np.diff(peaksTimes), np.nan])

                bNum = b_num(b_file, time)
                CheckPeaks = np.array(bNum).astype(np.float64)
                CheckPeaks[CheckPeaks != 2] = np.nan
                ToDelete = []
                for p in range(len(peaksLCFreq) - 1):

                    if np.sum(np.isnan(CheckPeaks[peaksLCFreq[p]:(peaksLCFreq[p + 1] + 1)])):
                        ToDelete.append(1)
                    else:
                        ToDelete.append(0)

                ToDelete.append(0)

                FreqCor = np.delete(Freq, np.array(ToDelete) == 1)
                PeakTimesCor = np.delete(peaksTimes, np.array(ToDelete) == 1)

                windowsBorders = EqualTimeSpentInNrem(0, b_file, 11)

                FreqWindowed = []
                WindowsTimes = []

                for w in windowsBorders:
                    start = w[0] * 4
                    end = w[1] * 4

                    WindowsTimes.append(np.mean([start, end]))

                    FreqWindow = FreqCor[np.logical_and(PeakTimesCor >= start, PeakTimesCor <= end)]

                    FreqMean = np.nanmean(FreqWindow)

                    FreqWindowed.append(FreqMean)

                key['lc_freq'] = np.array(FreqWindowed)
                key['window_times'] = np.array(WindowsTimes)

                self.insert1(key)
            else:
                print('No recording for this session')

def FCBehaviour_autopopulation(self, key):
    """
    FCBehaviour_autopopulation
    -------------
    Auto-populates processed neuroscience experiment data into a DataJoint table.

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. Perform task-specific preprocessing or calculations.
    3. Insert results into the database.

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    sessionType = (lcProj.FCSession & key).fetch1('session_type')

    FCDataPath = (os.sep + os.sep  + 'nasdcsr.unil.ch' + os.sep + 'RECHERCHE' + os.sep + 'FAC' + os.sep + 'FBM' + os.sep + 'DNF' + os.sep + 'aluthi1' + os.sep + 'sleep' + os.sep + 'D2c'
                     + os.sep + 'PROJECT_Georgios' + os.sep + 'LC_Proj_Neurons' + os.sep + 'Data'+ os.sep + 'FC' + os.sep + 'FCData' + os.sep)

    if sessionType in ['FCRecallAfter','FCRecallAfterExtAfter','FCExtinctionAfter']:

        mouseName = key['mouse_name']

        print(key)

        pathToMouse = FCDataPath + mouseName + os.sep

        # InfoDir = pathlib.Path(pathToMouse + 'ExperimentInfo.txt')
        #
        # Info = {}
        # with open(InfoDir) as f:
        #     for line in f:
        #         (keyDict, val) = line.split()
        #         Info[keyDict] = val

        #for keyDict in Info:

        year = key['session_date'][0:4]
        month = key['session_date'][5:7]
        day = key['session_date'][8:11]
        SessionDateFile = day + month + year

        pathToFCData = pathlib.Path(pathToMouse + os.sep + '*' + SessionDateFile)

        FCDataFolder = list(glob.glob(str(pathToFCData)))[0]

        FCDatapath = list(glob.glob(str(pathToFCData)))[0]

        FCDataFileSearch =  pathlib.Path(FCDatapath + os.sep + '*.txt')

        FCDataFile = list(glob.glob(str(FCDataFileSearch)))[0]

        fcData = []
        with open(FCDataFile, encoding='utf-16') as file:
            for line in file:
                fcData.append(line.strip().split(';'))

        Headers = fcData[0]
        HeadersCor = []

        for d in Headers[:-1]:
            HeadersCor.append(d.split('"')[1])


        FileName =  os.path.basename(FCDatapath)

        FileComponents = FileName.split('_')

        mousePos = {}
        Pos = []

        for pos, c in enumerate(FileComponents):

            if ('AMYG' in c) or ('MPC' in c):

                # if (mouseName in ['MPCFC02','MPCFC04']) and (key['session_date'] == '2025-03-28'):
                #
                #     mousePos[c] = pos + 1
                #     Pos.append(pos+1)
                #
                #     print('happened!')
                # else:

                mousePos[c] = pos
                Pos.append(pos)

        DataCor = []
        Data = fcData[1::]

        if (mouseName in ['MPCFC02','MPCFC04']) and (key['session_date'] == '2025-03-28'):
            posToLook = 1 + 3 * (np.max(Pos)+1) + mousePos[mouseName]
        else:
            posToLook =  1 + 3*np.max(Pos) + mousePos[mouseName]

        MouseFCData= []

        for d1 in Data[:-1]:
            for i, d2 in enumerate(d1[:-1]):
                if i == posToLook:
                    MouseFCData.append(float(d2))

        MouseFCData = np.array(MouseFCData)

        if (sessionType == 'FCRecallAfter') or (sessionType == 'FCRecallAfterExtAfter'):

            AbsstimTimes = [180, 30 + 120, 30 + 150, 30 + 120, 30 + 150]

            BaselinePeriod = []
            Stims = []

            mouseBas = []
            mouseStim = []
            for i, s in enumerate(AbsstimTimes):

                if i == 0:
                    currentStimStart = 0
                    currentStimEnd = int(np.sum(AbsstimTimes[0:(i + 1)]) / 10)
                    FreezingTimeBas = np.sum(MouseFCData[currentStimStart:currentStimEnd]) / (
                                (currentStimEnd - currentStimStart) * 10)
                    mouseBas.append(FreezingTimeBas * 100)
                    FreezingTime = np.sum(MouseFCData[currentStimEnd:(currentStimEnd + 3)]) / (3 * 10)
                    mouseStim.append(FreezingTime * 100)

                else:
                    currentStimStart = int(np.sum(AbsstimTimes[0:(i + 1)]) / 10)
                    currentStimEnd = int(currentStimStart + 30 / 10)
                    FreezingTime = np.sum(MouseFCData[currentStimStart:currentStimEnd]) / (
                                (currentStimEnd - currentStimStart) * 10)
                    mouseStim.append(FreezingTime * 100)

            BaselinePeriod.append(mouseBas)
            Stims.append(mouseStim)


            AllFreeze =  np.hstack([BaselinePeriod,np.array(Stims)])

        if (sessionType == 'FCExtinctionAfter'):

            AbsstimTimes = [180, 30 + 60, 30 + 70, 30 + 65, 30 + 67, 30 + 61, 30 + 60, 30 + 55, 30 + 63, 30 + 70,
                            30 + 60, 30 + 61, 30 + 62, 30 + 58, 30 + 71, 30 + 80, 30 + 64, 30 + 59, 30 + 67,
                            30 + 72, 30 + 80, 30 + 65, 30 + 60, 30 + 62, 30 + 64]

            BaselinePeriod = []
            Stims = []

            mouseBas = []
            mouseStim = []
            for i, s in enumerate(AbsstimTimes):

                if i == 0:
                    currentStimStart = 0
                    currentStimEnd = int(np.sum(AbsstimTimes[0:(i + 1)]) / 10)
                    FreezingTimeBas = np.sum(MouseFCData[currentStimStart:currentStimEnd]) / (
                            (currentStimEnd - currentStimStart) * 10)
                    mouseBas.append(FreezingTimeBas * 100)
                    FreezingTime = np.sum(MouseFCData[currentStimEnd:(currentStimEnd + 3)]) / (3 * 10)
                    mouseStim.append(FreezingTime * 100)

                else:
                    currentStimStart = int(np.sum(AbsstimTimes[0:(i + 1)]) / 10)
                    currentStimEnd = int(currentStimStart + 30 / 10)
                    FreezingTime = np.sum(MouseFCData[currentStimStart:currentStimEnd]) / (
                            (currentStimEnd - currentStimStart) * 10)
                    mouseStim.append(FreezingTime * 100)

            BaselinePeriod.append(mouseBas)
            Stims.append(mouseStim)

            AllFreeze =  np.hstack([BaselinePeriod,np.array(Stims)])



        key['freezing_time'] = MouseFCData
        key['freezing_per'] = AllFreeze.squeeze()
        self.insert1(key)


def DFFMeanEqualTimesNREM_autopopulation(self, key):
    """
    DFFMeanEqualTimesNREM_autopopulation
    -------------
    Auto-populates processed neuroscience experiment data into a DataJoint table.

    Workflow:
    ---------
    1. Fetch required raw data from the database.
    2. Perform task-specific preprocessing or calculations.
    3. Insert results into the database.

    Parameters
    ----------
    self : DataJoint table object
        Target table for data insertion.
    key : dict
        Primary key identifying the session.

    Returns
    -------
    None
        Inserts the processed data into the target table.
    """

    print('Computing mouse: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

    keyToCompute = (lcProj.Session & key).fetch()

    print(keyToCompute['project_id'])

    if ('SD' in  keyToCompute['project_id'][0]) or ('SSD' in keyToCompute['project_id'][0]):
        print('This is a deprivation session')

        dff = (lcProj.FibDFF & key).fetch1('dff')
        b_file = (lcProj.Hypnogram & key).fetch1('bfile')
        time = (lcProj.FibDFF & key).fetch1('timestamps')
        dff = dff.copy()
        fitbas = baseline(dff, time, 60, b_file)
        dffCor = dff - fitbas

        dffCor = zscore(dffCor)

        # find the first "n" or "2" in the file
        TransToNREM = re.finditer('[w1][n2]{1,}[w1n2r3]',b_file)

        TransSpans = []

        for idx, trans in enumerate(TransToNREM):
            Beg = list(trans.span())[0] + 1
            End = list(trans.span())[1] - 1
            TransSpans.append([Beg, End])

        TransTime = TransSpans[0][0]

        windowsBorders = EqualTimeSpentInNrem(TransTime, b_file, 8)

        WindowsTimes = []

        DffMeanNREMWindows = []

        for w in windowsBorders:
            start = w[0] * 4
            end = w[1] * 4

            bInW = b_file[w[0]:w[1]]

            NREMEpochs = re.finditer('[n2]{3,}', bInW)

            DffMeanNREMEpochs = []

            for idx, nrembout in enumerate(NREMEpochs):

                IndexTimeStart = start + int(nrembout.span()[0]) * 4
                IndexTimeEnd = start + int(nrembout.span()[1]) * 4

                BegDFF = findPosFromTimestamps(time, IndexTimeStart)
                EndDFF = findPosFromTimestamps(time, IndexTimeEnd)

                DffMeanNREMEpochs.append(np.mean(dffCor[BegDFF:EndDFF]))

            DffMeanNREMWindows.append(np.mean(DffMeanNREMEpochs))

            WindowsTimes.append(np.mean([start, end]))

        key['dff_mean'] = np.array(DffMeanNREMWindows)
        key['window_times'] = np.array(WindowsTimes)

        self.insert1(key)


    elif (keyToCompute['project_id'][0] == 'LC Photometry of Projection Neurons Baseline'):

        print('This is baseline session')
        dff = (lcProj.FibDFF & key).fetch1('dff')
        b_file = (lcProj.Hypnogram & key).fetch1('bfile')
        time = (lcProj.FibDFF & key).fetch1('timestamps')
        dff = dff.copy()
        fitbas = baseline(dff,time,60,b_file)
        dffCor = dff - fitbas

        dffCor = zscore(dffCor)

        windowsBorders = EqualTimeSpentInNrem(0, b_file, 12)

        WindowsTimes = []

        DffMeanNREMWindows = []

        for w in windowsBorders:
            start = w[0] * 4
            end = w[1] * 4

            bInW = b_file[w[0]:w[1]]

            NREMEpochs = re.finditer('[n2]{3,}', bInW)

            DffMeanNREMEpochs = []

            for idx, nrembout in enumerate(NREMEpochs):

                IndexTimeStart = start + int(nrembout.span()[0])*4
                IndexTimeEnd  = start + int(nrembout.span()[1])*4

                BegDFF = findPosFromTimestamps(time,IndexTimeStart)
                EndDFF = findPosFromTimestamps(time, IndexTimeEnd)

                DffMeanNREMEpochs.append(np.mean(dffCor[BegDFF:EndDFF]))

            DffMeanNREMWindows.append(np.mean(DffMeanNREMEpochs))

            WindowsTimes.append(np.mean([start, end]))

        key['dff_mean'] = np.array(DffMeanNREMWindows)
        key['window_times'] = np.array(WindowsTimes)

        self.insert1(key)

    elif (keyToCompute['project_id'][0] == 'LC Photometry of Projection Neurons combined with Fear Conditioning'):

        if not('Before' in ((lcProj.FCSession & key).fetch1('session_type'))):

            print('This is FC session')
            if lcProj.FibDFF & key:

                dff = (lcProj.FibDFF & key).fetch1('dff')
                b_file = (lcProj.Hypnogram & key).fetch1('bfile')
                time = (lcProj.FibDFF & key).fetch1('timestamps')
                dff = dff.copy()
                fitbas = baseline(dff, time, 60, b_file)
                dffCor = dff - fitbas

                dffCor = zscore(dffCor)

                windowsBorders = EqualTimeSpentInNrem(0, b_file, 11)

                WindowsTimes = []

                DffMeanNREMWindows = []

                for w in windowsBorders:
                    start = w[0] * 4
                    end = w[1] * 4

                    bInW = b_file[w[0]:w[1]]

                    NREMEpochs = re.finditer('[n2]{3,}', bInW)

                    DffMeanNREMEpochs = []

                    for idx, nrembout in enumerate(NREMEpochs):
                        IndexTimeStart = start + int(nrembout.span()[0]) * 4
                        IndexTimeEnd = start + + int(nrembout.span()[1]) * 4

                        BegDFF = findPosFromTimestamps(time, IndexTimeStart)
                        EndDFF = findPosFromTimestamps(time, IndexTimeEnd)

                        DffMeanNREMEpochs.append(np.mean(dffCor[BegDFF:EndDFF]))

                    DffMeanNREMWindows.append(np.mean(DffMeanNREMEpochs))

                    WindowsTimes.append(np.mean([start, end]))

                key['dff_mean'] = np.array(DffMeanNREMWindows)
                key['window_times'] = np.array(WindowsTimes)

                self.insert1(key)
