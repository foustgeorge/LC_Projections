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

def LFPGoodQuality_autopopulation(self, key):
    """
    LFPGoodQuality_autopopulation
    -------------
    Inserts quality annotations for LFP signals.

    Workflow:
    ---------
    1. Load quality annotation text files for each brain area (dCA1, mPFC, S1).
    2. Check if session date is listed as 'good quality' for the mouse.
    3. Insert quality flags into the database.

    Notes
    -----
    - Quality flags: 1 = good quality, 0 = poor quality.
    
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

        # function in order to insert info regarding the good quality of the LFP signals only for mice with have a projection neurons and LFP of a same brain area

        nasPathLCFiber = (os.sep + os.sep + 'nasdcsr.unil.ch' + os.sep + 'RECHERCHE' + os.sep + 'FAC' + os.sep + 'FBM' + os.sep + 'DNF' + os.sep + 'aluthi1' + os.sep + 'sleep' + os.sep + 'D2c' +
                    os.sep + '_PROJECTS' + os.sep + 'PROJECT_Georgios' + os.sep + 'LC_Proj_Neurons' + os.sep + 'Data' + os.sep + 'Ephys' + os.sep)

        InfoDirHipQual = pathlib.Path(nasPathLCFiber + os.sep + 'LFP_Quality_dCA1.txt')
        InfoDirPFCQual = pathlib.Path(nasPathLCFiber + os.sep + 'LFP_Quality_mPFC.txt')
        InfoDirS1Qual = pathlib.Path(nasPathLCFiber + os.sep + 'LFP_Quality_S1.txt')

        InfoMHIP = {}
        with open(InfoDirHipQual) as f:
            for line in f:
                elements = line.split()
                InfoMHIP[elements[0]] = elements[1:]

        InfoMPC = {}
        with open(InfoDirPFCQual) as f:
            for line in f:
                elements = line.split()
                InfoMPC[elements[0]] = elements[1:]

        InfoS1 = {}
        with open(InfoDirS1Qual) as f:
            for line in f:
                elements = line.split()
                InfoS1[elements[0]] = elements[1:]


        if key['mouse_name'] in InfoMHIP.keys():

            DatesOfGoodQuality = InfoMHIP[key['mouse_name']]
            keyDate = key['session_date']
            dateToCheck =  keyDate[2:4] + keyDate[5:7] + keyDate[8:10]

            if  dateToCheck in DatesOfGoodQuality:
                quality = 1
            else:
                quality = 0

            key['quality_hip'] = quality

        else:
            quality = 0
            key['quality_hip'] = quality




        if key['mouse_name'] in InfoMPC.keys():

            DatesOfGoodQuality = InfoMPC[key['mouse_name']]
            keyDate = key['session_date']
            dateToCheck = keyDate[2:4] + keyDate[5:7] + keyDate[8:10]

            if dateToCheck in DatesOfGoodQuality:
                quality = 1
            else:
                quality = 0

            key['quality_pfc'] = quality

        else:

            quality = 0
            key['quality_pfc'] = quality



        if key['mouse_name'] in InfoS1.keys():

            DatesOfGoodQuality = InfoS1[key['mouse_name']]
            keyDate = key['session_date']
            dateToCheck =  keyDate[2:4] + keyDate[5:7] + keyDate[8:10]

            if  dateToCheck in DatesOfGoodQuality:
                quality = 1
            else:
                quality = 0


            key['quality_s1'] = quality
        else:
            quality = 0
            key['quality_s1'] = quality



        self.insert1(key)

def REMLatency_autopopulation(self, key):
    """
    REMLatency_autopopulation
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

def StimPoints_autopopulation(self, key):
    """
    StimPoints_autopopulation
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

    # This a function to compute the time the LEDs were on during opto stimulation using the saved vector of stimulation the the RasPi sends to Intan

    if (lcProj.OptoSession & key):

        print('Inserting the stimulation points: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

        fileToRead = (lcProj.Session & key).fetch1('file_name')

        file = mat73.loadmat(fileToRead)

        # this is the saved vector
        if  key['mouse_name'] in ['NEJaws02','NEJaws03']:

            StimTrace = file['traces'][7,:]

        else:

            StimTrace = file['traces'][6,:]


        StimTraceBinary = np.empty(StimTrace.shape)

        #check when this vector is High
        for i, v in enumerate(StimTrace):

            if v > 0.95:
                StimTraceBinary[i] = 1
            else:
                StimTraceBinary[i] = 0

        Diff = np.diff(StimTraceBinary)

        OnTimes = np.where(Diff>0)[0]
        OffTimes = np.where(Diff<0)[0]

        stimCounter = 0

        #Check that each ON stimulation has an OFF if not exclude it (it's at the end of the file)

        if not(OnTimes.shape[0] == 0):

            if not(OnTimes.shape[0] == OffTimes.shape[0]):

                OnTimes = OnTimes[0:OffTimes.shape[0]]

                print('Stimulation start without stimulation end, excluded..')

            for i in range(OnTimes.shape[0]):

                stimCounter += 1

                OnTime = OnTimes[i]/1000
                try:
                    OffTime = OffTimes[i]/1000
                except:
                    OffTime = np.nan

                key['stim_num'] = stimCounter
                key['ontime'] = OnTime
                key['offtime'] = OffTime

                if  key['mouse_name'] in ['RT34','RT35','RT36'] and  str(key['session_date'])  ==  '2023-03-14':

                        print('End of sham stimulation corrected for 2h!')

                        OffTime = OnTime + 2*3600

                        key['offtime'] = OffTime

                                    # find the points on the Hypnogram where the ON and OFF timepoint fall
                if ((lcProj.Hypnogram & key).fetch1('bfile')) :

                    bfile = (lcProj.Hypnogram & key).fetch1('bfile')

                    TimeHyp = np.arange(0,len(bfile)*4,4)

                    TimeHypOnIndex = np.where(TimeHyp >= OnTime)[0][0]

                    TimeHypOffIndex = np.where(TimeHyp <= OffTime)[0][-1]

                    statesInStim = bfile[np.max([0,TimeHypOnIndex]):np.min([TimeHypOffIndex+1,len(bfile)])]

                    if statesInStim:
                        key['states_in_stim'] = statesInStim
                        self.insert1(key)
                        continue
                    else:
                        stimCounter -= 1
                        continue

                self.insert1(key)


def StimTimeSpent_autopopulation(self, key):
    """
    StimTimeSpent_autopopulation
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

    # This is a function to calculate time spent in different states during the 20 min of opto inhibition of
    # the LC for either the 20min as a whole or for smaller windows from the start of the stim

    if (lcProj.StimPoints & key).fetch1('states_in_stim'):

        print('Doing: ' + key['mouse_name'] + '---')
        print('Doing: ' + key['session_date'])
        print('Doing stim : ' + str(key['stim_num']))

        states = (lcProj.StimPoints & key).fetch1('states_in_stim')

        stimwindow = key['stimwindow']

        print('Analysing window : ' + str(key['stimwindow']))

        if stimwindow == 20:
            statesToAnalyse = states
        else:

            CharToKeep = int(np.ceil(stimwindow*60/4))
            statesToAnalyse = states[0:CharToKeep]

        # find the amount of different letters for n,r,w, 1,2,3 for their artifacts
        RemBouts1 = re.findall('r',statesToAnalyse)
        RemBouts2 = re.findall('3',statesToAnalyse)

        NoRemBouts1 = re.findall('n',statesToAnalyse)
        NoRemBouts2 = re.findall('2',statesToAnalyse)

        AwakeBouts1 = re.findall('w',statesToAnalyse)
        AwakeBouts2 = re.findall('1',statesToAnalyse)

        RemBoutsTotal = len(RemBouts1) +  len(RemBouts2)

        NoRemBoutsTotal =  len(NoRemBouts1) + len(NoRemBouts2)

        AwakeBoutsTotal =  len(AwakeBouts1) +  len(AwakeBouts2)

        timeInRem = RemBoutsTotal/len(statesToAnalyse)
        timeInNoRem = NoRemBoutsTotal/len(statesToAnalyse)
        timeInAwake = AwakeBoutsTotal/len(statesToAnalyse)

        key['time_spent_norem'] = timeInNoRem
        key['time_spent_rem'] = timeInRem
        key['time_spent_awake'] = timeInAwake

        window = key['window']

        print('Calculating: ' + key['mouse_name'] + ' for session: ' +
        (lcProj.Session & key).fetch1('session_date') + ', stim: ' + str(key['stim_num']) +
        ' and window at: ' + str(window)  + 's')

        RealTime = len(statesToAnalyse)*4

        #find the bouts of r,n and w (it looks complicated but it makes sure you get the bouts at the beginning and the ends of the file)
        rem = re.findall('[r*3*]*r+[3*r*]*|[r*3*]*3+[3*r*]*',statesToAnalyse)
        RemEpochs = []

        for r in rem :
            RemEpochs.append(len(r)*4)

        RemEpochsArray = np.array(RemEpochs)

        norem =re.findall('[n*2*]*n+[2*n*]*|[n*2*]*2+[2*n*]*',statesToAnalyse)
        NoRemEpochs = []
        for n in norem :
            NoRemEpochs.append(len(n)*4)

        NoRemEpochsArray = np.array(NoRemEpochs)

        awake = re.findall('[w*1*]*w+[1*w*]*|[w*1*]*1+[1*w*]*',statesToAnalyse)
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
        statesWindowed = wrap(states,WindowLengthInCharacters)

        RemWindowed = np.empty(len(statesWindowed))
        NoRemWindowed = np.empty(len(statesWindowed))
        AwakeWindowed = np.empty(len(statesWindowed))

        for i,c in enumerate(statesWindowed):

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

        self.insert1(key)

def StimTransitions_autopopulation(self, key):
    """
    StimTransitions_autopopulation
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

    # This is a function to calculate the transition in different states during the 20 min of opto inhibition of
    # the LC for either the 20min as a whole or for smaller windows from the start of the stim

    if (lcProj.StimPoints & key).fetch1('states_in_stim'):

        print('Doing: ' + key['mouse_name'] + '---')
        print('Doing: ' + key['session_date'])
        print('Doing stim : ' + str(key['stim_num']))

        states = (lcProj.StimPoints & key).fetch1('states_in_stim')

        window = key['window']

        stimwindow = key['stimwindow']

        WindowLengthInCharacters = int(window/4)

        print('Analysing window : ' + str(key['stimwindow']))

        if stimwindow == 20:
            statesToAnalyse = states
        else:
            CharToKeep = int(np.ceil(stimwindow*60/4))
            statesToAnalyse = states[0:CharToKeep]

        #find all the different types of transitions (the wake transitions are calculated after MA elimination

        RemTransitions1 = re.findall('nr',statesToAnalyse)
        RemTransitions2 = re.findall('n3',statesToAnalyse)
        RemTransitions3 = re.findall('2r',statesToAnalyse)
        RemTransitions4 = re.findall('23',statesToAnalyse)

        NREMBouts1 =  re.findall('n',statesToAnalyse)
        NREMBouts2 =  re.findall('2',statesToAnalyse)

        REMTransitionsTotal = len(RemTransitions1) + len(RemTransitions2) + len(RemTransitions3) + len(RemTransitions4)

        NREMTotalTime =  (len(NREMBouts1) + len( NREMBouts2))*4/60

        #remove the MA to count the transitions to wake
        [statesMACor, MASpans, PerMAs] = ElimMA(statesToAnalyse,12)

        # replace all the "m"s found in the vector with "0" and caclulate then the transitions to wake

        MAVector = ''
        for i in range(len(statesToAnalyse)):
            if np.sum(np.sum(MASpans)):
                if i in MASpans[:,0]:
                    MAVector = MAVector  + 'm'
                else:
                    MAVector = MAVector  + '0'
            else:
                MAVector = MAVector  + '0'

        MAWindowed = wrap(MAVector,WindowLengthInCharacters)


        MAAll = re.findall('m',MAVector)

        WakeTransitions1 = re.findall('nw',statesMACor)
        WakeTransitions2 = re.findall('n1',statesMACor)
        WakeTransitions3 = re.findall('2w',statesMACor)
        WakeTransitions4 = re.findall('21',statesMACor)

        NREMBouts1MACor =  re.findall('n',statesMACor)
        NREMBouts2MACor =  re.findall('2',statesMACor)

        WakeTransitionsTotal = len(WakeTransitions1) + len(WakeTransitions2) + len(WakeTransitions3) +len( WakeTransitions4)

        MATotal = len(MAAll)

        NREMTotalTimeMACor =  (len(NREMBouts1MACor) + len( NREMBouts2MACor))*4/60


        if  NREMTotalTime == 0:
            transitionsREM = np.nan
            microarousals = np.nan
        else:
            transitionsREM = REMTransitionsTotal/NREMTotalTime
            microarousals = MATotal/NREMTotalTime


        if  NREMTotalTimeMACor == 0:
            transitionsWake = np.nan
        else:
            transitionsWake = WakeTransitionsTotal/NREMTotalTimeMACor

        key['transitions_rem'] = transitionsREM
        key['transitions_awake'] = transitionsWake
        key['microarousals'] = microarousals


        print('Calculating: ' + key['mouse_name'] + ' for session: ' +
        (lcProj.Session & key).fetch1('session_date') + ', stim: ' + str(key['stim_num']) +
        ' and window at: ' + str(window)  + 's')


        # This analysis is in case we would like to window the stimulation window to get the dynamics


        WindowLengthInCharacters = int(window/4)

        # window in non-overlapping windows and repeat the calculation
        statesWindowed = wrap(states,WindowLengthInCharacters)
        statesWindowedMACor = wrap(statesMACor,WindowLengthInCharacters)

        TransRemWindowed = np.empty(len(statesWindowed))
        TransAwakeWindowed = np.empty(len(statesWindowedMACor))

        for i,c in enumerate(statesWindowed):

            RemTransitions1 = re.findall('nr',c)
            RemTransitions2 = re.findall('n3',c)
            RemTransitions3 = re.findall('2r',c)
            RemTransitions4 = re.findall('23',c)

            NREMBouts1 =  re.findall('n',c)
            NREMBouts2 =  re.findall('2',c)

            REMTransitionsTotal = len(RemTransitions1) + len(RemTransitions2) + len(RemTransitions3) + len(RemTransitions4)

            NREMTotalTime =  (len(NREMBouts1) + len( NREMBouts2))*4/60

            if  NREMTotalTime == 0:
                transitionsREMWin = np.nan
            else:
                transitionsREMWin = REMTransitionsTotal/NREMTotalTime

            TransRemWindowed[i] = transitionsREMWin

        for i,c in enumerate(statesWindowedMACor):

            WakeTransitions1 = re.findall('nw',c)
            WakeTransitions2 = re.findall('n1',c)
            WakeTransitions3 = re.findall('2w',c)
            WakeTransitions4 = re.findall('21',c)

            NREMBouts1MACor =  re.findall('n',c)
            NREMBouts2MACor  =  re.findall('2',c)

            WakeTransitionsTotal = len(WakeTransitions1) + len(WakeTransitions2) + len(WakeTransitions3) +len( WakeTransitions4)

            NREMTotalTimeMACor  =  (len(NREMBouts1MACor ) + len( NREMBouts2MACor ))*4/60

            if  NREMTotalTimeMACor  == 0:
                transitionsWakeWin = np.nan
            else:
                transitionsWakeWin = WakeTransitionsTotal/NREMTotalTimeMACor

            TransAwakeWindowed[i] = transitionsWakeWin

        NMAWindowed = np.empty(len(MAWindowed))

        for i,c in enumerate(zip(MAWindowed,statesWindowed)):

            MA = re.findall('m',c[0])
            NREMBouts1 =  re.findall('n',c[1])
            NREMBouts2 =  re.findall('2',c[1])

            MATotal = len(MA)
            NREMTotalTime =  (len(NREMBouts1) + len( NREMBouts2))*4/60

            if  NREMTotalTime  == 0:
                MAWin = np.nan
            else:
                MAWin = MATotal/NREMTotalTime

            NMAWindowed[i] = MAWin

        key['transitions_rem_windowed'] = TransRemWindowed
        key['transitions_awake_windowed'] = TransAwakeWindowed
        key['microarousals_windowed'] = NMAWindowed

        self.insert1(key)


def SSDMAOpto_autopopulation(self, key):
    """
    SSDMAOpto_autopopulation
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

    #This is a function to compute the density of MA after SSD paired with LC inhibition

    if lcProj.StimPoints & key:

        # get the b-file which is between the stim points

        bInStim = (lcProj.StimPoints & key).fetch1('states_in_stim')

        # this ElimMA fucntion returns the spans of all the MAs detected

        [statesMACor, MASpans, PerMAs] = ElimMA(bInStim,12)

        MAs =  MASpans.shape[0]

        NREMBouts1 =  re.findall('n',bInStim )
        NREMBouts2 =  re.findall('2',bInStim )

        # calculate the total time in NREM

        NREMTotalTime =  (len(NREMBouts1) + len( NREMBouts2))*4/60

        MADensity = MAs/NREMTotalTime

        sessionType = (lcProj.OptoSession & key).fetch1('session_type')

        key['ma_density'] = MADensity
        key['session_type'] = sessionType

        self.insert1(key)

def MADensity_autopopulation(self, key):
    """
    MADensity_autopopulation
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

    # This is a function to calculate the density of MA after SD or SSD for an hour after the first detected NREM

    if ('SD' in (lcProj.Session() & key).fetch('project_id')[0]) and not((lcProj.OptoSession & key)) and (not ('REMSD' in (lcProj.Session() & key).fetch('project_id')[0])):

            print('Calculating MA Density ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

            bfile = (lcProj.Hypnogram & key).fetch1('bfile')

            # find the first "n" or "2" in the file
            TransToNREM = re.finditer('[w1][n2]{1,}[w1n2r3]',bfile)

            TransSpans = []

            for idx, trans in enumerate(TransToNREM):

                    Beg = list(trans.span())[0]+1
                    End = list(trans.span())[1]-1
                    TransSpans.append([Beg,End])

            TransTime = TransSpans[0][0]

            bfileOneHour = bfile[(TransTime) : (TransTime + 900)]

            # this ElimMA fucntion returns the spans of all the MAs detected

            [statesMACor, MASpans, PerMAs] = ElimMA(bfileOneHour,12)

            # this gives the number of MAs
            MAs =  MASpans.shape[0]

            NREMBouts1 =  re.findall('n',bfileOneHour)
            NREMBouts2 =  re.findall('2',bfileOneHour)

            NREMTotalTime =  (len(NREMBouts1) + len( NREMBouts2))*4/60

            MADensity = MAs/NREMTotalTime

            if lcProj.ImagingSession & key :

                sessionType = (lcProj.ImagingSession & key).fetch1('session_type')

            elif lcProj.REMSession & key:

                sessionType = (lcProj.REMSDSession & key).fetch1('session_type')

            key['ma_density'] = MADensity
            key['session_info'] = sessionType

            self.insert1(key)


def DeltaDynamics_autopopulation(self, key):
    """
    DeltaDynamics_autopopulation
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

def StimTransitions_autopopulation(self, key):
    """
    StimTransitions_autopopulation
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

    # This is a function to calculate the transition in different states during the 20 min of opto inhibition of
    # the LC for either the 20min as a whole or for smaller windows from the start of the stim

    if (lcProj.StimPoints & key).fetch1('states_in_stim'):

        print('Doing: ' + key['mouse_name'] + '---')
        print('Doing: ' + key['session_date'])
        print('Doing stim : ' + str(key['stim_num']))

        states = (lcProj.StimPoints & key).fetch1('states_in_stim')

        window = key['window']

        stimwindow = key['stimwindow']

        WindowLengthInCharacters = int(window/4)

        print('Analysing window : ' + str(key['stimwindow']))

        if stimwindow == 20:
            statesToAnalyse = states
        else:
            CharToKeep = int(np.ceil(stimwindow*60/4))
            statesToAnalyse = states[0:CharToKeep]

        #find all the different types of transitions (the wake transitions are calculated after MA elimination

        RemTransitions1 = re.findall('nr',statesToAnalyse)
        RemTransitions2 = re.findall('n3',statesToAnalyse)
        RemTransitions3 = re.findall('2r',statesToAnalyse)
        RemTransitions4 = re.findall('23',statesToAnalyse)

        NREMBouts1 =  re.findall('n',statesToAnalyse)
        NREMBouts2 =  re.findall('2',statesToAnalyse)

        REMTransitionsTotal = len(RemTransitions1) + len(RemTransitions2) + len(RemTransitions3) + len(RemTransitions4)

        NREMTotalTime =  (len(NREMBouts1) + len( NREMBouts2))*4/60

        #remove the MA to count the transitions to wake
        [statesMACor, MASpans, PerMAs] = ElimMA(statesToAnalyse,12)

        # replace all the "m"s found in the vector with "0" and caclulate then the transitions to wake

        MAVector = ''
        for i in range(len(statesToAnalyse)):
            if np.sum(np.sum(MASpans)):
                if i in MASpans[:,0]:
                    MAVector = MAVector  + 'm'
                else:
                    MAVector = MAVector  + '0'
            else:
                MAVector = MAVector  + '0'

        MAWindowed = wrap(MAVector,WindowLengthInCharacters)


        MAAll = re.findall('m',MAVector)

        WakeTransitions1 = re.findall('nw',statesMACor)
        WakeTransitions2 = re.findall('n1',statesMACor)
        WakeTransitions3 = re.findall('2w',statesMACor)
        WakeTransitions4 = re.findall('21',statesMACor)

        NREMBouts1MACor =  re.findall('n',statesMACor)
        NREMBouts2MACor =  re.findall('2',statesMACor)

        WakeTransitionsTotal = len(WakeTransitions1) + len(WakeTransitions2) + len(WakeTransitions3) +len( WakeTransitions4)

        MATotal = len(MAAll)

        NREMTotalTimeMACor =  (len(NREMBouts1MACor) + len( NREMBouts2MACor))*4/60


        if  NREMTotalTime == 0:
            transitionsREM = np.nan
            microarousals = np.nan
        else:
            transitionsREM = REMTransitionsTotal/NREMTotalTime
            microarousals = MATotal/NREMTotalTime


        if  NREMTotalTimeMACor == 0:
            transitionsWake = np.nan
        else:
            transitionsWake = WakeTransitionsTotal/NREMTotalTimeMACor

        key['transitions_rem'] = transitionsREM
        key['transitions_awake'] = transitionsWake
        key['microarousals'] = microarousals


        print('Calculating: ' + key['mouse_name'] + ' for session: ' +
        (lcProj.Session & key).fetch1('session_date') + ', stim: ' + str(key['stim_num']) +
        ' and window at: ' + str(window)  + 's')


        # This analysis is in case we would like to window the stimulation window to get the dynamics


        WindowLengthInCharacters = int(window/4)

        # window in non-overlapping windows and repeat the calculation
        statesWindowed = wrap(states,WindowLengthInCharacters)
        statesWindowedMACor = wrap(statesMACor,WindowLengthInCharacters)

        TransRemWindowed = np.empty(len(statesWindowed))
        TransAwakeWindowed = np.empty(len(statesWindowedMACor))

        for i,c in enumerate(statesWindowed):

            RemTransitions1 = re.findall('nr',c)
            RemTransitions2 = re.findall('n3',c)
            RemTransitions3 = re.findall('2r',c)
            RemTransitions4 = re.findall('23',c)

            NREMBouts1 =  re.findall('n',c)
            NREMBouts2 =  re.findall('2',c)

            REMTransitionsTotal = len(RemTransitions1) + len(RemTransitions2) + len(RemTransitions3) + len(RemTransitions4)

            NREMTotalTime =  (len(NREMBouts1) + len( NREMBouts2))*4/60

            if  NREMTotalTime == 0:
                transitionsREMWin = np.nan
            else:
                transitionsREMWin = REMTransitionsTotal/NREMTotalTime

            TransRemWindowed[i] = transitionsREMWin

        for i,c in enumerate(statesWindowedMACor):

            WakeTransitions1 = re.findall('nw',c)
            WakeTransitions2 = re.findall('n1',c)
            WakeTransitions3 = re.findall('2w',c)
            WakeTransitions4 = re.findall('21',c)

            NREMBouts1MACor =  re.findall('n',c)
            NREMBouts2MACor  =  re.findall('2',c)

            WakeTransitionsTotal = len(WakeTransitions1) + len(WakeTransitions2) + len(WakeTransitions3) +len( WakeTransitions4)

            NREMTotalTimeMACor  =  (len(NREMBouts1MACor ) + len( NREMBouts2MACor ))*4/60

            if  NREMTotalTimeMACor  == 0:
                transitionsWakeWin = np.nan
            else:
                transitionsWakeWin = WakeTransitionsTotal/NREMTotalTimeMACor

            TransAwakeWindowed[i] = transitionsWakeWin

        NMAWindowed = np.empty(len(MAWindowed))

        for i,c in enumerate(zip(MAWindowed,statesWindowed)):

            MA = re.findall('m',c[0])
            NREMBouts1 =  re.findall('n',c[1])
            NREMBouts2 =  re.findall('2',c[1])

            MATotal = len(MA)
            NREMTotalTime =  (len(NREMBouts1) + len( NREMBouts2))*4/60

            if  NREMTotalTime  == 0:
                MAWin = np.nan
            else:
                MAWin = MATotal/NREMTotalTime

            NMAWindowed[i] = MAWin

        key['transitions_rem_windowed'] = TransRemWindowed
        key['transitions_awake_windowed'] = TransAwakeWindowed
        key['microarousals_windowed'] = NMAWindowed

        self.insert1(key)



def InterREMIntervals_autopopulation(self, key):
    """
    InterREMIntervals_autopopulation
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


    if (lcProj.REMSDSession & key).fetch('session_type') == "REM sleep deprivation":

        print('Calculating inter-REM interval ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

        bfile = (lcProj.Hypnogram & key).fetch1('bfile')

        WakeMiddle = re.finditer('[n2][w1]{4,}[n2]',bfile)

        WakeEnd = re.finditer('[n2][w1]{4,}' + '$',bfile)

        WakeBeg= re.finditer('^' + '[w1]{4,}[n2]',bfile)

        WakeAfterREM = re.finditer('[r3][w1]{4,}[n2]',bfile)

        WakeAfterREMEnd = re.finditer('[r3][w1]{4,}' + '$',bfile)

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

        bfileWakeCor = bfile

        for i, w in enumerate(WakeSpansArray):

            Temp = bfileWakeCor[0:w[0]] + '0' * (w[1]-w[0])  + bfileWakeCor[w[1]:]

            bfileWakeCor = Temp

        NREMBouts = []
        HourOfScoring = int(3600/4)
        bREMSD = bfileWakeCor[0:(HourOfScoring*6)]

        seg = int(len(bREMSD)/HourOfScoring)
        InterREMBoutsBsHours = []

        for i in range(seg):

            bHour = bREMSD[(i*HourOfScoring):((i+1)*HourOfScoring)]

            #InterREMBouts = re.findall('[0w1n2]{1,}[n2]{3,}',bHour)

            InterREMBouts = re.finditer('[0w1n2]{1,}',bHour)

            InterREMBoutsBs = []

            for idx, w in enumerate(InterREMBouts):

                Beg = list(w.span())[0]
                End = list(w.span())[1]

                if not(End == len(bHour))  and  (bHour[End] == 'r'):
                    InterREMBoutsBs.append(bHour[Beg:End])

            InterREMBoutsBsHours.append(InterREMBoutsBs)

        BoutLengthMean = []
        BoutLengthAll = []

        for i, h in enumerate(InterREMBoutsBsHours):

            BoutHourLength = []

            for n in h:

                Ns=  re.findall('n',n)
                Twos  =  re.findall('2',n)
                Ws  =  re.findall('w',n)
                Ones  =  re.findall('1',n)

                BoutHourLength.append((len(Ns) + len(Twos) + len(Ws) + len(Ones))*4)



            BoutLengthMean.append(np.mean(BoutHourLength))
            BoutLengthAll.append(BoutHourLength)

        key['inter_rem_interval_mean'] = BoutLengthMean
        key['inter_rem_interval_all'] = BoutLengthAll

        self.insert1(key)

def LCMinREMSD_autopopulation(self, key):
    """
    LCMinREMSD_autopopulation
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

    #load the PSA caclulated by MATLAB consolidated

    PSAPath = (os.sep + os.sep  + 'nasdcsr.unil.ch' + os.sep + 'RECHERCHE' + os.sep + 'FAC' + os.sep + 'FBM' + os.sep + 'DNF' + os.sep + 'aluthi1' + os.sep + 'sleep' + os.sep + 'D2c' +
                     os.sep + 'PROJECT_LC_ultradian_regulation' + os.sep + 'General Scripts' + os.sep + 'PSA_Fig1.mat')
    mouseName = key['mouse_name']

    if 'CI' in mouseName:

        print('Doing mouse: ' + mouseName)

        mouseData = mat73.loadmat(PSAPath, only_include = 'PSA/' + mouseName)

        sessionInfo = (lcProj.Session & key).fetch1('file_name')



        if 'REMSD1' in sessionInfo:

            try:

                LCminData =   mouseData['PSA'][mouseName]['Rec_6hREMSD1_REMSDLCMin']['LCmin']

                if LCminData.shape[0] == 2:

                    LCminData =  np.nan * np.ones(12)


                key['lc_min_activity'] = LCminData

                self.insert1(key)

            except:

                print('Session excluded due to low DR')


        elif 'REMSD2' in sessionInfo:

            try:

                LCminData =    mouseData['PSA'][mouseName]['Rec_6hREMSD2_REMSDLCMin']['LCmin']

                if LCminData.shape[0] == 2:

                    LCminData = np.nan * np.ones(12)

                key['lc_min_activity'] = LCminData

                self.insert1(key)

            except:

                print('Session excluded due to low DR')


def REMSDMotorTimes_autopopulation(self, key):
    """
    REMSDMotorTimes_autopopulation
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

        if 'CI' in key['mouse_name'] :

            print('Computing mouse: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

            fileToRead = (lcProj.Session & key).fetch1('file_name')

            file = mat73.loadmat(fileToRead)

            if key['mouse_name'] in ['CI13','CI14']:

                MotorTrace = file['traces'][8,:][0:6*3600*1000]
            else:

                MotorTrace = file['traces'][10,:][0:6*3600*1000]

            bf, af = signal.cheby2(6, 40, 30/500, btype ='low', analog=False, output='ba')

            MotorTracefilt = signal.filtfilt(bf, af, MotorTrace)

            MotorTraceBinary = np.empty(MotorTrace.shape)

            #check when this vector is High
            for i, v in enumerate(MotorTracefilt):

                if v >1.2:
                    MotorTraceBinary[i] = 1
                elif v < 0.7:
                    MotorTraceBinary[i] = 0

            Diff = np.diff(MotorTraceBinary)

            OnTimes = np.array(np.where(Diff>0)[0])/1000

            key['motor_times_on'] = OnTimes

            self.insert1(key)

def REMBeforeAndWakerAfterMotor_autopopulation(self,key):
    """
    REMBeforeAndWakerAfterMotor_autopopulation
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

    if 'CI' in key['mouse_name'] :

        print('Computing mouse: ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

        print('Computing time in REM before motor..')


        OnTime = (lcProj.REMSDMotorTimes & key).fetch1('motor_times_on')*1000
        bfile = (lcProj.Hypnogram & key).fetch1('bfile')

        HypAllTimes = ''

        for i, b in enumerate(bfile):

            HypAllTimes = HypAllTimes + b*4*1000


        rbegAll = []
        rendAll = []

        for t in OnTime:


            if  HypAllTimes[int(t)] == 'r' or  HypAllTimes[int(t)] == '3':

                REMEndFound = 1

                rend = int(t)

                rendAll.append(rend)

                REMBegFound = 0

                rbeg = rend

                while not(REMBegFound):

                    rbeg  = rbeg  - 1

                    if not(HypAllTimes[rbeg] == 'r' or HypAllTimes[rbeg] == '3'):

                        REMBegFound= 1

                rbegAll.append(rbeg)

            else:

                REMEndFound = 0

                rend = int(t)

                while not(REMEndFound):

                    rend  = rend  - 1

                    if HypAllTimes[rend] == 'r' or HypAllTimes[ rend] == '3':

                        REMEndFound = 1

                rendAll.append(rend)

                REMBegFound = 0

                rbeg = rend

                while not(REMBegFound):

                    rbeg  = rbeg  - 1

                    if not(HypAllTimes[rbeg] == 'r' or HypAllTimes[rbeg] == '3'):

                        REMBegFound= 1

                rbegAll.append(rbeg)


        TimesFloatREM = np.abs((np.array(rbegAll) - np.array(rendAll)- np.ones(len(rendAll)))/1000)

        TimesREM  = [int(i) for i in TimesFloatREM ]

        key['rem_before'] = np.mean(TimesREM)

        print('Computing time in Wake after motor..')

        wbegAll = []
        wendAll = []

        for t in OnTime:

            if  HypAllTimes[int(t)] == 'w' or  HypAllTimes[int(t)] == '1':

                WakBegFound = 1

                wbeg = int(t)

                wbegAll.append(wbeg)

                WakeEndFound = 0

                wend = wbeg

                while not(WakeEndFound):

                    wend  = wend   + 1

                    if not(HypAllTimes[wend] == 'w' or HypAllTimes[wend] == '1'):

                        WakeEndFound= 1

                wendAll.append(wend)

            else:

                WakeBegFound = 0

                wbeg = int(t)

                while not(WakeBegFound):

                    wbeg  = wbeg  + 1

                    if HypAllTimes[wbeg] == 'w' or HypAllTimes[wbeg] == '1':

                        WakeBegFound = 1

                wbegAll.append(wbeg)

                WakeEndFound = 0

                wend = wbeg

                while not(WakeEndFound):

                    wend  = wend   + 1

                    if not(HypAllTimes[wend] == 'w' or HypAllTimes[wend] == '1'):

                        WakeEndFound= 1

                wendAll.append(wend)


        TimesFloatWake = np.abs((np.array(wbegAll) - np.array(wendAll)- np.ones(len(wendAll)))/1000)

        TimesWake  = [int(i) for i in TimesFloatWake]

        key['wake_after'] = np.mean(TimesWake)

        self.insert1(key)

def SSDREMBoutsOpto_autopopulation(self, key):
    """
    SSDREMBoutsOpto_autopopulation
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

    #This is a function to compute the density of MA after SSD paired with LC inhibition

    if lcProj.StimPoints & key:

        # get the b-file which is between the stim points

        bInStim = (lcProj.StimPoints & key).fetch1('states_in_stim')

        # this ElimMA fucntion returns the spans of all the MAs detected

        rem = re.findall('[r*3*]*r+[3*r*]*|[r*3*]*3+[3*r*]*',bInStim)
        RemEpochs = []

        for r in rem :
            RemEpochs.append(len(r)*4)

        RemEpochsArray = np.array(RemEpochs)

        sessionType = (lcProj.OptoSession & key).fetch1('session_type')


        RemBouts1 = re.findall('r',bInStim)
        RemBouts2 = re.findall('3',bInStim)

        RemBoutsTotal = len(RemBouts1) +  len(RemBouts2)

        timeInRem = RemBoutsTotal/len(bInStim)

        RemTransitions1 = re.findall('nr',bInStim)
        RemTransitions2 = re.findall('n3',bInStim)
        RemTransitions3 = re.findall('2r',bInStim)
        RemTransitions4 = re.findall('23',bInStim)

        NREMBouts1 =  re.findall('n',bInStim)
        NREMBouts2 =  re.findall('2',bInStim)

        REMTransitionsTotal = len(RemTransitions1) + len(RemTransitions2) + len(RemTransitions3) + len(RemTransitions4)
        NREMTotalTime =  (len(NREMBouts1) + len( NREMBouts2))*4/60


        transitionsREM = REMTransitionsTotal/NREMTotalTime


        key['time_spent_rem'] = timeInRem
        key['transitions_rem'] = transitionsREM
        key['rem_bouts'] = np.mean(RemEpochsArray)
        key['session_type'] = sessionType

        self.insert1(key)

def REMInfoSDvsSSD_autopopulation(self, key):
    """
    REMInfoSDvsSSD_autopopulation
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

    if ('SD' in (lcProj.Session() & key).fetch('project_id')[0]) and not((lcProj.OptoSession & key)) and (not ('REMSD' in (lcProj.Session() & key).fetch('project_id')[0])):

            print('Calculating MA Density ' + key['mouse_name'] + ' for session: ' + (lcProj.Session & key).fetch1('session_date'))

            bfile = (lcProj.Hypnogram & key).fetch1('bfile')

            # find the first "n" or "2" in the file
            TransToNREM = re.finditer('[w1][n2]{1,}[w1n2r3]',bfile)

            TransSpans = []

            for idx, trans in enumerate(TransToNREM):

                    Beg = list(trans.span())[0]+1
                    End = list(trans.span())[1]-1
                    TransSpans.append([Beg,End])

            TransTime = TransSpans[0][0]

            bfileOneHour = bfile[(TransTime) : (TransTime + 900)]

            rem = re.findall('[r*3*]*r+[3*r*]*|[r*3*]*3+[3*r*]*',bfileOneHour)
            RemEpochs = []

            for r in rem :
                RemEpochs.append(len(r)*4)

            RemEpochsArray = np.array(RemEpochs)


            RemBouts1 = re.findall('r',bfileOneHour)
            RemBouts2 = re.findall('3',bfileOneHour)

            RemBoutsTotal = len(RemBouts1) +  len(RemBouts2)

            timeInRem = RemBoutsTotal/len(bfileOneHour)

            RemTransitions1 = re.findall('nr',bfileOneHour)
            RemTransitions2 = re.findall('n3',bfileOneHour)
            RemTransitions3 = re.findall('2r',bfileOneHour)
            RemTransitions4 = re.findall('23',bfileOneHour)

            NREMBouts1 =  re.findall('n',bfileOneHour)
            NREMBouts2 =  re.findall('2',bfileOneHour)

            REMTransitionsTotal = len(RemTransitions1) + len(RemTransitions2) + len(RemTransitions3) + len(RemTransitions4)
            NREMTotalTime =  (len(NREMBouts1) + len( NREMBouts2))*4/60


            transitionsREM = REMTransitionsTotal/NREMTotalTime

            if lcProj.ImagingSession & key :

                sessionType = (lcProj.ImagingSession & key).fetch1('session_type')

            elif lcProj.REMSession & key:

                sessionType = (lcProj.REMSDSession & key).fetch1('session_type')


            key['time_spent_rem'] = timeInRem
            key['transitions_rem'] = transitionsREM
            key['rem_bouts'] = np.mean(RemEpochsArray)
            key['session_info'] = sessionType

            self.insert1(key)

def LCActivitySDvsSSD_autopopulation(self, key):
    """
    LCActivitySDvsSSD_autopopulation
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

        sessionType = ['Bas','SSD','SD']

        if key['mouse_name'] in ['CI03','CI05','CI06','CI07','CI10','CI13','CI14']:

            print('Importing LC activity ' + key['mouse_name'])

            for s in sessionType:

                LCActivityPath = (os.sep + os.sep  + 'nasdcsr.unil.ch' + os.sep + 'RECHERCHE' + os.sep + 'FAC' + os.sep + 'FBM' + os.sep + 'DNF' + os.sep + 'aluthi1' + os.sep + 'sleep' + os.sep + 'D2c' +
                                 os.sep + 'PROJECT_LC_ultradian_regulation' + os.sep + 'Folders_per_Experiments' + os.sep + 'LC_CalciumImaging' + os.sep + 'RawData' + os.sep + key['mouse_name'] + os.sep + 'LCFreq' + os.sep )

                keyMouse = {'mouse_name':key['mouse_name']}

                if s == 'SD':

                    LCActivityPathSD =  LCActivityPath  + 'FreqSD.txt'

                    lines = []
                    with open(LCActivityPathSD, 'r') as f:
                        for line in f:
                            lines.append(line.strip().split())

                    LCActivity = [float(i) for i in lines[0]]
                    Times = [float(i) for i in lines[1]]

                    key['session_info'] = 'Sleep Deprivation'

                    key['lc_activity'] = LCActivity

                    key['times'] = Times

                    self.insert1(key)


                elif  s == 'SSD':

                    LCActivityPathSSD =  LCActivityPath  + 'FreqSSD.txt'


                    lines = []
                    with open(LCActivityPathSSD, 'r') as f:
                        for line in f:
                            lines.append(line.strip().split())

                    LCActivity = [float(i) for i in lines[0]]
                    Times = [float(i) for i in lines[1]]

                    key['session_info'] = 'Sensory Enchanced Sleep Deprivation'

                    key['lc_activity'] = LCActivity

                    key['times'] = Times

                    self.insert1(key)


                elif s == 'Bas':

                    LCActivityPathBas =  LCActivityPath  + 'FreqBas.txt'

                    lines = []
                    with open(LCActivityPathBas, 'r') as f:
                        for line in f:
                            lines.append(line.strip().split())

                    LCActivity = [float(i) for i in lines[0]]
                    Times = [float(i) for i in lines[1]]

                    key['session_info'] = 'Baseline Session'

                    key['lc_activity'] = LCActivity

                    key['times'] = Times

                    self.insert1(key)

def TimeSleepInOpto_autopopulation(self, key):
    """
    TimeSleepInOpto_autopopulation
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

    #This is a function to compute the density of MA after SSD paired with LC inhibition

    if lcProj.StimPoints & key:

        # get the b-file which is between the stim points

        bInStim = (lcProj.StimPoints & key).fetch1('states_in_stim')

        # this ElimMA fucntion returns the spans of all the MAs detected

        NREMBouts1 =  re.findall('n',bInStim )
        NREMBouts2 =  re.findall('2',bInStim )

        REMBouts1 =  re.findall('r',bInStim )
        REMBouts2 =  re.findall('3',bInStim )

        # calculate the total time in NREM

        NREMTotalTime =  (len(NREMBouts1) + len( NREMBouts2))*4/60
        REMTotalTime =  (len(REMBouts1) + len( REMBouts2))*4/60

        SleepTotalTime = (NREMTotalTime + REMTotalTime)/(len(bInStim)*4/3600)

        sessionType = (lcProj.OptoSession & key).fetch1('session_type')

        key['time_in_sleep'] = SleepTotalTime
        key['session_type'] = sessionType

        self.insert1(key)


def  DFFSleepStates_autopopulation(self, key):
    """
    DFFSleepStates_autopopulation
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
    """
    safe_mean_concat
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

def NremRemMeanDFFSSD_autopopulation(self, key):
    """
    NremRemMeanDFFSSD_autopopulation
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

    bfile = (lcProj.Hypnogram & key).fetch1('bfile')
    dff = (lcProj.FibDFF & key).fetch1('dff')
    timestamps = (lcProj.FibDFF & key).fetch1('timestamps')

    # Detect the first "n" (or 1) in the b file
    TransToNREM = re.finditer('[w1][n2]{1,}[w1n2r3]',bfile)

    TransSpans = []

    for idx, trans in enumerate(TransToNREM):

            Beg = list(trans.span())[0]+1
            End = list(trans.span())[1]-1
            TransSpans.append([Beg,End])

    TransTime = TransSpans[0][0]

    After3h = int(TransSpans[0][0] + 3*3600/4)

    bAfterTrans = bfile[TransTime:After3h]

    InterREMbouts = reg.finditer('[r3]{3,}', bAfterTrans)

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

        bInBetween = bAfterTrans[StartOne:EndOne]

        CurrentRemLength = len(re.findall('[r3]',bAfterTrans[PointStart[i]:PointEnd[i]]))*4

        NremInBetween = re.finditer('[n2]{1,}', bInBetween)

        NremInBetweenFlag = len(re.findall('[n2]{1,}', bInBetween))

        if NremInBetweenFlag > 0:

            BegDFF =  (StartOne + TransSpans[0][0])  * 4

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

def LFPSignals_autopopulation(self, key):
    """
    LFPSignals_autopopulation
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

    if (lcProj.LFPGoodQuality & key).fetch1('quality_hip'):
        LFP_Hip = traces[5, :]

        # one mouse with different configuration
        if ((key['mouse_name'] == 'MPC05') and (key['session_date'] == "2023-08-07")) or ((key['mouse_name'] == 'THAL01') and (key['session_date'] == "2023-08-01")):

            print('Different configuration')
            LFP_Hip = traces[4, :]

        LFP_Hip = signal.resample(LFP_Hip, int(LFP_Hip.shape[0] / 5))
        key['lfp_hip'] =  LFP_Hip
        print('Hip injected')

    else:
        LFP_Hip = np.zeros([traces[5, :].shape[0]])*np.nan
        LFP_Hip = signal.resample(LFP_Hip, int(LFP_Hip.shape[0] / 5))
        key['lfp_hip'] = LFP_Hip
        print('dCA1 bad quality')

    if (lcProj.LFPGoodQuality & key).fetch1('quality_s1'):
        LFP_S1 = traces[4, :]

            # one mouse with different configuration
        if ((key['mouse_name'] == 'MPC05') and (key['session_date'] == "2023-08-07")) or ((key['mouse_name'] == 'THAL01') and (key['session_date'] == "2023-08-01")):

            print('Different configuration')
            LFP_S1 = traces[5, :]

        LFP_S1 = signal.resample(LFP_S1, int(LFP_S1.shape[0] / 5))
        key['lfp_s1'] = LFP_S1
        print('S1 injected')
    else:
        LFP_S1 = np.zeros([traces[4, :].shape[0]])*np.nan
        LFP_S1 = signal.resample(LFP_S1, int(LFP_S1.shape[0] / 5))
        key['lfp_s1'] = LFP_S1
        print('S1 bad quality')


    if (lcProj.LFPGoodQuality & key).fetch1('quality_pfc'):
        LFP_PFC = traces[6, :]
            # one mouse with different configuration
        if ((key['mouse_name'] == 'MPC05') and (key['session_date'] == "2023-08-07")) or ((key['mouse_name'] == 'THAL01') and (key['session_date'] == "2023-08-01")):

            print('Different configuration')
            LFP_PFC = traces[6, :]

        LFP_PFC = signal.resample(LFP_PFC, int(LFP_PFC.shape[0] / 5))
        key['lfp_pfc'] = LFP_PFC
        print('mPFC injected')
    else:
        LFP_PFC = np.zeros([traces[6, :].shape[0]])*np.nan
        LFP_PFC = signal.resample(LFP_PFC, int(LFP_PFC.shape[0] / 5))
        key['lfp_pfc'] = LFP_PFC
        print('mPFC bad quality')

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

def MGTHipLFP_autopopulation(self, key):
    """
    MGTHipLFP_autopopulation
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

    if (lcProj.LFPGoodQuality & key).fetch1('quality_hip'):
        #fileName = (lcProj.Session & key).fetch1('file_name')

        #print('Loading traces---')
        #matFile = mat73.loadmat(fileName)
        #traces = matFile['traces']

        #LFP_Hip = traces[5,:]

        # one mouse with different configuration
        #if ((key['mouse_name']  == 'MPC05') and (key['session_date'] == "2023-08-07")) or  ((key['mouse_name']  == 'THAL01') and (key['session_date'] == "2023-08-01")):
        #    print('Different configuration')
        #    LFP_Hip = traces[4,:]

        #LFP_Hip = signal.resample(LFP_Hip, int(LFP_Hip.shape[0]/5))

        LFP_Hip = (lcProj.LFPSignals & key).fetch1('lfp_hip')

        print('Computing the Hip electrode')
        print('Computing delta---')
        WTDelta_Hip = MGT(LFP_Hip, 200, 1.5, 4, 0.25)
        WTDelta_Hip= np.abs(WTDelta_Hip)
        WTDelta_Hip = np.mean(WTDelta_Hip, 0)
        WTDelta_Hip = signal.resample(WTDelta_Hip, int(WTDelta_Hip.shape[0] / 2))

        print('Computing sigma---')
        WTSigma_Hip = MGT(LFP_Hip, 200, 10, 15, 0.25)
        WTSigma_Hip = np.abs(WTSigma_Hip)
        WTSigma_Hip= np.mean(WTSigma_Hip, 0)
        WTSigma_Hip = signal.resample(WTSigma_Hip, int(WTSigma_Hip.shape[0] / 2))

        print('Computing gamma---')
        WTGamma_Hip = MGT(LFP_Hip, 200, 60, 80, 0.25)
        WTGamma_Hip = np.abs(WTGamma_Hip)
        WTGamma_Hip = np.mean(WTGamma_Hip, 0)
        WTGamma_Hip = signal.resample(WTGamma_Hip, int(WTGamma_Hip.shape[0] / 2))

        key['delta_hip'] = WTDelta_Hip
        key['sigma_hip'] = WTSigma_Hip
        key['gamma_hip'] = WTGamma_Hip

        self.insert1(key)
    else:
        print('Bad quality LFP, skipped')


def MGTS1LFP_autopopulation(self, key):
    """
    MGTS1LFP_autopopulation
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

    if (lcProj.LFPGoodQuality & key).fetch1('quality_s1'):
        #fileName = (lcProj.Session & key).fetch1('file_name')

        #print('Loading traces---')
        #matFile = mat73.loadmat(fileName)
        #traces = matFile['traces']

        #LFP_S1 = traces[4, :]

        # one mouse with different configuration
        #if ((key['mouse_name']  == 'MPC05') and (key['session_date'] == "2023-08-07")) or  ((key['mouse_name']  == 'THAL01') and (key['session_date'] == "2023-08-01")):
        #    print('Different configuration')
        #    LFP_S1 = traces[5, :]

        #LFP_S1= signal.resample(LFP_S1, int(LFP_S1.shape[0] / 5))

        LFP_S1 = (lcProj.LFPSignals & key).fetch1('lfp_s1')

        print('Computing the S1 electrode')
        print('Computing delta---')
        WTDelta_S1 = MGT(LFP_S1, 200, 1.5, 4, 0.25)
        WTDelta_S1= np.abs(WTDelta_S1)
        WTDelta_S1 = np.mean(WTDelta_S1, 0)
        WTDelta_S1 = signal.resample(WTDelta_S1, int(WTDelta_S1.shape[0] / 2))

        print('Computing sigma---')
        WTSigma_S1 = MGT(LFP_S1, 200, 10, 15, 0.25)
        WTSigma_S1 = np.abs(WTSigma_S1)
        WTSigma_S1= np.mean(WTSigma_S1, 0)
        WTSigma_S1 = signal.resample(WTSigma_S1, int(WTSigma_S1.shape[0] / 2))

        print('Computing gamma---')
        WTGamma_S1 = MGT(LFP_S1, 200, 60, 80, 0.25)
        WTGamma_S1 = np.abs(WTGamma_S1)
        WTGamma_S1 = np.mean(WTGamma_S1, 0)
        WTGamma_S1 = signal.resample(WTGamma_S1, int(WTGamma_S1.shape[0] / 2))

        key['delta_s1'] = WTDelta_S1
        key['sigma_s1'] = WTSigma_S1
        key['gamma_s1'] = WTGamma_S1

        self.insert1(key)
    else:
        print('Bad quality LFP, skipped')


def MGTPfcLFP_autopopulation(self, key):
    """
    MGTPfcLFP_autopopulation
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

    if (lcProj.LFPGoodQuality & key).fetch1('quality_pfc'):
    #    fileName = (lcProj.Session & key).fetch1('file_name')

    #    print('Loading traces---')
    #    matFile = mat73.loadmat(fileName)
    #    traces = matFile['traces']

    #    LFP_Pfc= traces[6,:]

        # one mouse with different configuration
    #    if ((key['mouse_name']  == 'MPC05') and (key['session_date'] == "2023-08-07")) or  ((key['mouse_name']  == 'THAL01') and (key['session_date'] == "2023-08-01")):
    #        print('Different configuration')

    #        LFP_Pfc = traces[6, :]

    #    LFP_Pfc = signal.resample(LFP_Pfc, int(LFP_Pfc.shape[0]/5))

        LFP_Pfc = (lcProj.LFPSignals & key).fetch1('lfp_pfc')

        print('Computing the Pfc electrode')
        print('Computing delta---')
        WTDelta_Pfc = MGT(LFP_Pfc, 200, 1.5, 4, 0.25)
        WTDelta_Pfc= np.abs(WTDelta_Pfc)
        WTDelta_Pfc = np.mean(WTDelta_Pfc, 0)
        WTDelta_Pfc = signal.resample(WTDelta_Pfc, int(WTDelta_Pfc.shape[0] / 2))

        print('Computing sigma---')
        WTSigma_Pfc = MGT(LFP_Pfc, 200, 10, 15, 0.25)
        WTSigma_Pfc = np.abs(WTSigma_Pfc)
        WTSigma_Pfc= np.mean(WTSigma_Pfc, 0)
        WTSigma_Pfc = signal.resample(WTSigma_Pfc, int(WTSigma_Pfc.shape[0] / 2))

        print('Computing gamma---')
        WTGamma_Pfc = MGT(LFP_Pfc, 200, 60, 80, 0.25)
        WTGamma_Pfc = np.abs(WTGamma_Pfc)
        WTGamma_Pfc = np.mean(WTGamma_Pfc, 0)
        WTGamma_Pfc = signal.resample(WTGamma_Pfc, int(WTGamma_Pfc.shape[0] / 2))

        key['delta_pfc'] = WTDelta_Pfc
        key['sigma_pfc'] = WTSigma_Pfc
        key['gamma_pfc'] = WTGamma_Pfc

        self.insert1(key)
    else:
        print('Bad quality LFP, skipped')


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





