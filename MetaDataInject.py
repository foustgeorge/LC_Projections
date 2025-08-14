# ============================================================
#  LC Projection Neurons Data Injection Pipeline
#  ------------------------------------------------------------
#  Automates ingestion of mouse metadata and experimental session
#  details into the DataJoint lcProj schema.
#
#  Author: Georgios Foustoukos
#  Date: 14.08.2025
#  ------------------------------------------------------------
#  Key Skills Demonstrated:
#    • Batch CSV parsing & ingestion to relational DB (DataJoint)
#    • Automated experiment-type detection
#    • NAS file system traversal & parsing
#    • Error handling for missing/incomplete datasets
#    • Support for multiple experimental paradigms:
#        - Baseline
#        - Sleep Deprivation (SD)
#        - Sensory-enhanced Sleep Deprivation (SSD)
#        - Fear Conditioning (FC)
# ============================================================

import os
from tqdm import tqdm
import pathlib
import requests
from Schema import lcProj
import matplotlib.pyplot as plt
import random
import pickle
import glob
import re
import pandas as pd
from skimage import io
import numpy as np
import mat73
import datetime
from datetime import datetime



def inject_mice_csv_batch():
    """
    Batch-ingests mouse metadata into the lcProj.Mouse table.

    Purpose:
    --------
    Automates reading CSV metadata files describing experimental mice, 
    validates required fields, replaces missing values with 'NoData', 
    and inserts structured records into the DataJoint database.

    Key Steps:
    ----------
    1. Locate `MiceMetaData` directory containing files matching `*LCProj.csv`.
    2. Read each file into a pandas DataFrame with robust NaN handling.
    3. Validate `MouseName` entries and skip records with missing names.
    4. Insert each cleaned metadata entry into the `lcProj.Mouse` table,
       avoiding duplicates.

    Error Handling:
    ---------------
    • Raises FileNotFoundError if metadata directory is missing.
    • Skips over records without a valid `MouseName`.
    """

    #Injection of mice
    #set the meta data dirs

    meta_data_dir= os.path.join(os.getcwd(), 'MiceMetaData')
    meta_data_dir = pathlib.Path(meta_data_dir)
    print("****")
    print(meta_data_dir)

    if not meta_data_dir.exists():
        raise FileNotFoundError(f'Path not found!! {meta_data_dir.as_posix()}')

    #  Read all the mouse info from the .csv file
    meta_data_files = meta_data_dir.glob('*LCProj.csv')

    for meta_data_file in tqdm(meta_data_files):

        print(meta_data_file)

        infodata =  pd.read_csv(meta_data_file, sep = ';')

        for lines in range(len(infodata)):

            if type(infodata['MouseName'][lines]) == str:
                    print("Everything is fine.")
            elif np.isnan(infodata['MouseName'][lines]):
                    print("The mouse has no name!")
                    continue

            MouseInfo={}    # make sure function robust
            MouseInfo = {
            'mouse_name': infodata['MouseName'][lines] if not pd.isnull(infodata['MouseName'][lines]) else 'NoData',
            'responsible_experimenter':infodata['ResponsibleExperimenter'][lines] if not pd.isnull(infodata['ResponsibleExperimenter'][lines]) else 'NoData',
            'pyrat_id': infodata['PyratID'][lines] if not pd.isnull(infodata['PyratID'][lines]) else 'NoData',
            'genotype': infodata['Genotype'][lines] if not pd.isnull(infodata['Genotype'][lines]) else 'NoData',
            'gender': infodata['Gender'][lines] if not pd.isnull(infodata['Gender'][lines]) else 'NoData',
            'birth_date': infodata['DateOfBirth'][lines] if not pd.isnull(infodata['DateOfBirth'][lines]) else 'NoData',
            'licence': infodata['LicenceAutorisation'][lines] if not pd.isnull(infodata['LicenceAutorisation'][lines]) else 'NoData',
            'dg': infodata['DegreeOfSeverity'][lines] if not pd.isnull(infodata['DegreeOfSeverity'][lines]) else 'NoData',
            'start_date': infodata['DateOfStart'][lines] if not pd.isnull(infodata['DateOfStart'][lines]) else 'NoData',
            'age': infodata['AgesInWeeks'][lines] if not pd.isnull(infodata['AgesInWeeks'][lines]) else 'NoData',
            'sacrifice_date': infodata['DateOfSacrifice'][lines] if not pd.isnull(infodata['DateOfSacrifice'][lines]) else 'NoData',
            'proj_type': infodata['ProjType'][lines] if not pd.isnull(infodata['ProjType'][lines]) else 'NoData',
            'filename' : os.path.split(meta_data_file)[-1]
            }
            print(MouseInfo)

            lcProj.Mouse.insert1(MouseInfo,skip_duplicates=True, ignore_extra_fields=True)
            print(MouseInfo['mouse_name'] + ' has been ingested.' )



def inject_sessions():
    """
    Ingests experimental session records into lcProj based on NAS-stored session data.

    Purpose:
    --------
    Scans experiment data directories for each registered mouse, determines 
    the type of experiment performed, extracts relevant metadata, and inserts 
    structured records into the corresponding DataJoint tables.

    Features:
    ---------
    • Detects multiple experiment types:
        - Baseline
        - Sleep Deprivation (SD)
        - Sensory-enhanced SD (SSD)
        - Fear Conditioning (FC)
    • Dynamically determines NAS path location for each mouse's data.
    • Parses `ExperimentInfo.txt` for experiment schedule.
    • Extracts recording time from MATLAB `.mat` scored files when available.
    • Handles multiple scored file scenarios to avoid duplicate ingestion.

    Error Handling:
    ---------------
    • Skips missing scored files and uses default times if unavailable.
    • Uses experimenter initials from `lcProj.Person` table for traceability.

    Notes:
    ------
    This function is a core ETL step, converting raw NAS file structure into 
    relational database records for downstream analysis.
    """


    #function to inject sessions depending on their data

    #Fetch lc the mice from DJ
    MiceNames = lcProj.Mouse.fetch('mouse_name')

    # set the NAS paths

    nasPathLCFiber= (os.sep + os.sep  + 'nasdcsr.unil.ch' + os.sep + 'RECHERCHE' + os.sep + 'FAC' + os.sep + 'FBM' + os.sep + 'DNF' + os.sep + 'aluthi1' + os.sep + 'sleep' + os.sep + 'D2c'
                     + os.sep + 'PROJECT_Georgios' + os.sep + 'LC_Proj_Neurons' + os.sep + 'Data'+ os.sep + 'Ephys' + os.sep)

    nasPathLCFiberFC= (os.sep + os.sep  + 'nasdcsr.unil.ch' + os.sep + 'RECHERCHE' + os.sep + 'FAC' + os.sep + 'FBM' + os.sep + 'DNF' + os.sep + 'aluthi1' + os.sep + 'sleep' + os.sep + 'D2c'
                     + os.sep + 'PROJECT_Georgios' + os.sep + 'LC_Proj_Neurons' + os.sep + 'Data'+ os.sep +'FC'+ os.sep + 'Ephys' + os.sep)


    for m in tqdm(MiceNames):

        LCDirFiber = pathlib.Path(nasPathLCFiber + os.sep + m)

        LCDirFiberFC = pathlib.Path(nasPathLCFiberFC + os.sep + m)

        if LCDirFiber.exists():
            DataPath = nasPathLCFiber
        else:
            DataPath = nasPathLCFiberFC

        if LCDirFiber.exists() or LCDirFiberFC.exists():

            # This is for the mice of fiber photometry

            print('This is a mouse from the LC and fiber photometry data with projection neurons!')

            print('----------Mouse: '+ m + '----------')

            key = {'mouse_name':m}

            ExperimenterNameQuery = 'full_name=' + '"' + (lcProj.Mouse & key).fetch1('responsible_experimenter') + '"'

            experimenter = (lcProj.Person & ExperimenterNameQuery).fetch1('initials')

            print('Loading sessions for mouse: '+ m)

            InfoDir = pathlib.Path(DataPath + m +'/ExperimentInfo.txt')

            Info = {}
            with open(InfoDir) as f:
                for line in f:
                    (key, val) = line.split()
                    Info[key] = val


            sessions = []
            sessionsDate = []

            for key in Info:

                print(key)

                if len(key) == 7:
                    keyCor = key[0:-1]
                else:
                    keyCor = key

                year = '20' + ''.join(keyCor[0:2])
                month = ''.join(keyCor[2:4])
                day = ''.join(keyCor[4:6])
                SessionDate = year + '-' + month + '-' + day

                keyCheck = {'mouse_name':m,
                            'session_date':SessionDate}


                SessionDateFile = year[2:4] +  month + day


                if Info[key] in ['BL1Fiber','BL2Fiber','BL3Fiber', 'BL4Fiber','BL5Fiber']:

                    projectId = 'LC Photometry of Projection Neurons Baseline'
                    rigRoom = 'Neurobau1.27'
                    sessionType1 = 'LC Calcium Fiber Photometry '
                    sessionType2 = 'Baseline Session'
                    baselineDuration = '12h'
                    areaFiberPhotometry= 'LC'
                    areaGampInjection = 'LC'
                    gcampVirusInfo = r'AAV5.hSyn1.dlox.GCaMP8s.dlox.WPRE-SV40p(A)'
                    retroVirusInfo = r'ssAAV-retro/2-mTH-iCre-WPRE-bGHp(A)'
                    gcampInjectionCoordinates = 'AP : -5.4 , ML: +0.9, DV:-3.2 to -2.2'
                    fiberAreaImplantation = 'LC'
                    fiberImplantationCoordinates = 'AP : -5.4 , ML: +0.9, DV:-2.8 '
                    powerLed = '12.5uW, 25uW'
                    ledWavelength = '405nm, 465nm'

                    ProjInfo = (lcProj.Mouse & keyCheck).fetch1('proj_type')

                    if ProjInfo == 'dCA1':
                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -2.46, ML: +2.0, DV:-1.27 '

                    elif ProjInfo == 'mPFC':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : +1.7 , ML: +0.3, DV:-2.75 to -2 '

                    elif ProjInfo == 'Thal':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -1.6 , ML: +0.95, DV:-3.0 '

                    elif ProjInfo == 'Amy':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -1.8 , ML: +3.2, DV:-3.5 '

                    dataPathScored = pathlib.Path(DataPath + m +'/' + SessionDateFile  + '*bt.mat')

                    scoredFile = list(glob.glob(str(dataPathScored)))[0]

                    matFile = mat73.loadmat(scoredFile,only_include='Infos')

                    if 'RecordingDate' in matFile['Infos'].keys():

                        dateTime =  datetime.strptime(matFile['Infos']['RecordingDate'],"%d-%b-%Y %H:%M:%S")

                        SessionTime = str(dateTime)[11::]

                    else:

                        SessionTime = '09:00:00'

                    filename = scoredFile

                    FibSessionDict = {  'mouse_name' : m,
                                        'session_date': SessionDate,
                                        'session_time' : SessionTime,
                                        'session_type': sessionType1,
                                        'area_fib' : areaFiberPhotometry,
                                        'gcamp_virus_info' : gcampVirusInfo,
                                        'retro_virus_info' : retroVirusInfo,
                                        'gcamp_injection_coordinates' : gcampInjectionCoordinates ,
                                        'retro_injection_coordinates': retroInjectionCoordinates,
                                        'area_fib_implantation': fiberAreaImplantation,
                                        'implantation_fib_coordinates':  fiberImplantationCoordinates,
                                        'power_led': powerLed,
                                        'led_wavelength': ledWavelength}


                    BaselineSessionDict = { 'mouse_name' : m,
                                            'session_date': SessionDate,
                                            'session_time' : SessionTime,
                                            'session_type': sessionType2,
                                            'baseline_duration' : baselineDuration
                                    }


                    lcProj.FibSession.insert1(FibSessionDict,skip_duplicates=True)
                    lcProj.BaselineSession.insert1(BaselineSessionDict ,skip_duplicates=True)



                    sessionDict = {  'mouse_name' : m,
                                     'session_date': SessionDate,
                                     'session_time' : SessionTime,
                                     'project_id' : projectId,
                                     'initials' : experimenter,
                                     'rig_room' : rigRoom,
                                     'file_name': filename
                                    }

                    lcProj.Session.insert1(sessionDict,skip_duplicates=True)


                if Info[key] in ['SDFiber']:

                    projectId = 'LC Photometry of Projection Neurons with SD'
                    rigRoom = 'Neurobau1.27'
                    sessionType1 = 'LC Calcium Fiber Photometry '
                    sessionType2 = 'Sleep Deprivation'
                    sdDuration = '4h'
                    areaFiberPhotometry= 'LC'
                    areaGampInjection = 'LC'
                    gcampVirusInfo = r'AAV5.hSyn1.dlox.GCaMP8s.dlox.WPRE-SV40p(A)'
                    retroVirusInfo = r'ssAAV-retro/2-mTH-iCre-WPRE-bGHp(A)'
                    gcampInjectionCoordinates = 'AP : -5.4 , ML: +0.9, DV:-3.2 to -2.2'
                    fiberAreaImplantation = 'LC'
                    fiberImplantationCoordinates = 'AP : -5.4 , ML: +0.9, DV:-2.8 '
                    powerLed = '15uW, 20uW'
                    ledWavelength = '405nm, 465nm'

                    ProjInfo = (lcProj.Mouse & keyCheck).fetch1('proj_type')

                    if ProjInfo == 'dCA1':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -2.46, ML: +2.0, DV:-1.27 '

                    elif ProjInfo == 'mPFC':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : +1.7 , ML: +0.3, DV:-2.75 to -2 '

                    elif ProjInfo == 'Thal':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -1.6 , ML: +0.95, DV:-3.0 '


                    elif ProjInfo == 'Amy':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -1.8 , ML: +3.2, DV:-3.5 '

                    dataPathScored = pathlib.Path(nasPathLCFiber + m +'/' + SessionDateFile  + '*bt.mat')

                    scoredFile = list(glob.glob(str(dataPathScored)))[0]

                    matFile = mat73.loadmat(scoredFile, only_include='Infos')

                    if 'RecordingDate' in matFile['Infos'].keys():

                        dateTime =  datetime.strptime(matFile['Infos']['RecordingDate'],"%d-%b-%Y %H:%M:%S")

                        SessionTime = str(dateTime)[11::]

                    else:

                        SessionTime = '09:00:00'

                    filename = scoredFile

                    FibSessionDict = {  'mouse_name' : m,
                                        'session_date': SessionDate,
                                        'session_time' : SessionTime,
                                        'session_type': sessionType1,
                                        'area_fib' : areaFiberPhotometry,
                                        'gcamp_virus_info' : gcampVirusInfo,
                                        'retro_virus_info' : retroVirusInfo,
                                        'gcamp_injection_coordinates' : gcampInjectionCoordinates ,
                                        'retro_injection_coordinates': retroInjectionCoordinates,
                                        'area_fib_implantation': fiberAreaImplantation,
                                        'implantation_fib_coordinates':  fiberImplantationCoordinates,
                                        'power_led': powerLed,
                                        'led_wavelength': ledWavelength}


                    sdSessionDict = {  'mouse_name' : m,
                                       'session_date': SessionDate,
                                       'session_time' : SessionTime,
                                       'session_type': sessionType2,
                                       'sd_duration' : sdDuration
                                    }


                    lcProj.FibSession.insert1(FibSessionDict,skip_duplicates=True)
                    lcProj.SDSession.insert1(sdSessionDict ,skip_duplicates=True)


                    sessionDict = {  'mouse_name' : m,
                                     'session_date': SessionDate,
                                     'session_time' : SessionTime,
                                     'project_id' : projectId,
                                     'initials' : experimenter,
                                     'rig_room' : rigRoom,
                                     'file_name': filename
                                    }

                    lcProj.Session.insert1(sessionDict,skip_duplicates=True)

                elif Info[key] in ['SSDFiber']:

                    projectId = 'LC Photometry of Projection Neurons with SSD'
                    rigRoom = 'Neurobau1.27'
                    sessionType1 = 'LC Calcium Fiber Photometry '
                    sessionType2 = 'Sensory Enchanced Sleep Deprivation'
                    ssdDuration = '4h'
                    areaFiberPhotometry= 'LC'
                    areaGampInjection = 'LC'
                    gcampVirusInfo = r'AAV5.hSyn1.dlox.GCaMP8s.dlox.WPRE-SV40p(A)'
                    retroVirusInfo = r'ssAAV-retro/2-mTH-iCre-WPRE-bGHp(A)'
                    gcampInjectionCoordinates = 'AP : -5.4 , ML: +0.9, DV:-3.2 to -2.2'
                    fiberAreaImplantation = 'LC'
                    fiberImplantationCoordinates = 'AP : -5.4 , ML: +0.9, DV:-2.8 '
                    powerLed = '15uW, 20uW'
                    ledWavelength = '405nm, 465nm'

                    ProjInfo = (lcProj.Mouse & keyCheck).fetch1('proj_type')

                    if ProjInfo == 'dCA1':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -2.46, ML: +2.0, DV:-1.27 '

                    elif ProjInfo == 'mPFC':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : +1.7 , ML: +0.3, DV:-2.75 to -2 '

                    elif ProjInfo == 'Thal':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -1.6 , ML: +0.95, DV:-3.0 '

                    elif ProjInfo == 'Amy':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -1.8 , ML: +3.2, DV:-3.5 '

                    dataPathScored = pathlib.Path(DataPath+ m +'/' + SessionDateFile + '*bt.mat')

                    scoredFile = list(glob.glob(str(dataPathScored)))[0]

                    matFile = mat73.loadmat(scoredFile,only_include='Infos')

                    if 'RecordingDate' in matFile['Infos'].keys():

                        dateTime =  datetime.strptime(matFile['Infos']['RecordingDate'],"%d-%b-%Y %H:%M:%S")

                        SessionTime = str(dateTime)[11::]

                    else:

                        SessionTime = '09:00:00'

                    filename = scoredFile


                    FibSessionDict = {  'mouse_name' : m,
                                        'session_date': SessionDate,
                                        'session_time' : SessionTime,
                                        'session_type': sessionType1,
                                        'area_fib' : areaFiberPhotometry,
                                        'gcamp_virus_info' : gcampVirusInfo,
                                        'retro_virus_info' : retroVirusInfo,
                                        'gcamp_injection_coordinates' : gcampInjectionCoordinates ,
                                        'retro_injection_coordinates': retroInjectionCoordinates,
                                        'area_fib_implantation': fiberAreaImplantation,
                                        'implantation_fib_coordinates':  fiberImplantationCoordinates,
                                        'power_led': powerLed,
                                        'led_wavelength': ledWavelength}


                    ssdSessionDict = {       'mouse_name' : m,
                                             'session_date': SessionDate,
                                             'session_time' : SessionTime,
                                             'session_type': sessionType2,
                                             'ssd_duration' : ssdDuration
                                    }



                    lcProj.FibSession.insert1(FibSessionDict,skip_duplicates=True)
                    lcProj.SSDSession.insert1(ssdSessionDict,skip_duplicates=True)



                    sessionDict = {  'mouse_name' : m,
                                     'session_date': SessionDate,
                                     'session_time' : SessionTime,
                                     'project_id' : projectId,
                                     'initials' : experimenter,
                                     'rig_room' : rigRoom,
                                     'file_name': filename
                                    }

                    lcProj.Session.insert1(sessionDict,skip_duplicates=True)


                elif Info[key] in ['FCTrainingBefore','FCTrainingAfter','FCRecallBefore','FCRecallAfter','FCExtinctionBefore','FCExtinctionAfter','FCRecallAfterExtBefore','FCRecallAfterExtAfter']:

                    projectId = 'LC Photometry of Projection Neurons combined with Fear Conditioning'
                    rigRoom = 'Neurobau1.22'
                    sessionType1 = 'LC Calcium Fiber Photometry '
                    sessionType2 = Info[key]
                    areaFiberPhotometry= 'LC'
                    areaGampInjection = 'LC'
                    gcampVirusInfo = r'AAV5.hSyn1.dlox.GCaMP8s.dlox.WPRE-SV40p(A)'
                    retroVirusInfo = r'ssAAV-retro/2-mTH-iCre-WPRE-bGHp(A)'
                    gcampInjectionCoordinates = 'AP : -5.4 , ML: +0.9, DV:-3.2 to -2.2'
                    fiberAreaImplantation = 'LC'
                    fiberImplantationCoordinates = 'AP : -5.4 , ML: +0.9, DV:-2.8 '
                    powerLed = '15uW, 20uW'
                    ledWavelength = '405nm, 465nm'

                    if Info[key] in ['FCTrainingBefore','FCTrainingAfter']:

                            stimuliNumber = 5
                            footShockIntensity = 0.5
                            footShockDuration = 2
                            soundIntensity = 80
                            soundFrequency = 1000
                            soundDuration = 30
                            soundFootShockOverlap = 'last 2 sec'

                    elif Info[key] in ['FCRecallBefore','FCRecallAfter','FCRecallAfterExtBefore','FCRecallAfterExtAfter']:

                            stimuliNumber = 5
                            footShockIntensity = 0
                            footShockDuration = 0
                            soundIntensity = 80
                            soundFrequency = 1000
                            soundDuration = 30
                            soundFootShockOverlap = 'None'

                    elif Info[key] in ['FCExtinctionBefore','FCExtinctionAfter',]:

                            stimuliNumber = 25
                            footShockIntensity = 0
                            footShockDuration = 0
                            soundIntensity = 80
                            soundFrequency = 1000
                            soundDuration = 30
                            soundFootShockOverlap = 'None'

                    ProjInfo = (lcProj.Mouse & keyCheck).fetch1('proj_type')

                    if ProjInfo == 'dCA1':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -2.46, ML: +2.0, DV:-1.27 '

                    elif ProjInfo == 'mPFC':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : +1.7 , ML: +0.3, DV:-2.75 to -2 '

                    elif ProjInfo == 'Thal':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -1.6 , ML: +0.95, DV:-3.0 '

                    elif ProjInfo == 'Amy':

                        areaRetroInjection = ProjInfo
                        retroInjectionCoordinates = 'AP : -1.8 , ML: +3.2, DV:-3.5 '

                    dataPathScored = pathlib.Path(DataPath+ m +'/' + SessionDateFile + '*bt.mat')

                    scoredFileCheck = list(glob.glob(str(dataPathScored)))

                    if not (len(scoredFileCheck) == 0):

                        if (len(scoredFileCheck) == 1):

                            scoredFile = list(glob.glob(str(dataPathScored)))[0]

                        else:
                            if lcProj.Session & keyCheck:
                                scoredFile = list(glob.glob(str(dataPathScored)))[0]
                            else:
                                scoredFile = list(glob.glob(str(dataPathScored)))[1]

                    if not (len(scoredFileCheck) == 0):

                        matFile = mat73.loadmat(scoredFile,only_include='Infos')

                        if 'RecordingDate' in matFile['Infos'].keys():

                            dateTime =  datetime.strptime(matFile['Infos']['RecordingDate'],"%d-%b-%Y %H:%M:%S")

                            SessionTime = str(dateTime)[11::]

                        else:

                            SessionTime = '09:00:00'

                        filename = scoredFile


                        FibSessionDict = {  'mouse_name' : m,
                                            'session_date': SessionDate,
                                            'session_time' : SessionTime,
                                            'session_type': sessionType1,
                                            'area_fib' : areaFiberPhotometry,
                                            'gcamp_virus_info' : gcampVirusInfo,
                                            'retro_virus_info' : retroVirusInfo,
                                            'gcamp_injection_coordinates' : gcampInjectionCoordinates ,
                                            'retro_injection_coordinates': retroInjectionCoordinates,
                                            'area_fib_implantation': fiberAreaImplantation,
                                            'implantation_fib_coordinates':  fiberImplantationCoordinates,
                                            'power_led': powerLed,
                                            'led_wavelength': ledWavelength}


                        fcSessionDict = {'mouse_name' : m,
                                         'session_date': SessionDate,
                                         'session_time' : SessionTime,
                                         'session_type': sessionType2,
                                         'stim_number' : stimuliNumber,
                                         'footshock_int': footShockIntensity,
                                         'footshock_dur': footShockDuration,
                                         'sound_int' : soundIntensity,
                                         'sound_dur': soundDuration,
                                         'sound_freq': soundFrequency,
                                         'sound_footshock_overlap': soundFootShockOverlap
                                        }



                        lcProj.FibSession.insert1(FibSessionDict,skip_duplicates=True)
                        lcProj.FCSession.insert1(fcSessionDict, skip_duplicates=True)
                        sessionDict = {'mouse_name': m,
                                       'session_date': SessionDate,
                                       'session_time': SessionTime,
                                       'project_id': projectId,
                                       'initials': experimenter,
                                       'rig_room': rigRoom,
                                       'file_name': filename
                                       }

                        lcProj.Session.insert1(sessionDict, skip_duplicates=True)

                    else:

                        if 'Before' in Info[key]:

                            SessionTime = '09:00:00'
                        else:
                            SessionTime = '11:00:00'

                        filename = 'No Photometry Data'
                        fcSessionDict = {'mouse_name' : m,
                                         'session_date': SessionDate,
                                         'session_time' : SessionTime,
                                         'session_type': sessionType2,
                                         'stim_number' : stimuliNumber,
                                         'footshock_int': footShockIntensity,
                                         'footshock_dur': footShockDuration,
                                         'sound_int' : soundIntensity,
                                         'sound_dur': soundDuration,
                                         'sound_freq': soundFrequency,
                                         'sound_footshock_overlap': soundFootShockOverlap
                                        }

                        lcProj.FCSession.insert1(fcSessionDict,skip_duplicates=True)



                        sessionDict = {  'mouse_name' : m,
                                         'session_date': SessionDate,
                                         'session_time' : SessionTime,
                                         'project_id' : projectId,
                                         'initials' : experimenter,
                                         'rig_room' : rigRoom,
                                         'file_name': filename
                                        }

                        lcProj.Session.insert1(sessionDict,skip_duplicates=True)
