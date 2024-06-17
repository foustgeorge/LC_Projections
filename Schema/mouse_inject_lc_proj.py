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

    #function to inject sessions depending on their data

    #Fetch lc the mice from DJ
    MiceNames = lcProj.Mouse.fetch('mouse_name')

    # set the NAS paths

    nasPathLCFiber= (os.sep + os.sep  + 'nasdcsr.unil.ch' + os.sep + 'RECHERCHE' + os.sep + 'FAC' + os.sep + 'FBM' + os.sep + 'DNF' + os.sep + 'aluthi1' + os.sep + 'fbm_move' + os.sep + 'D2c' +
                                  os.sep + '_PROJECTS' + os.sep + 'PROJECT_Georgios' + os.sep + 'LC_Proj_Neurons' + os.sep + 'Ephys' + os.sep)
    for m in tqdm(MiceNames):

        LCDirFiber = pathlib.Path(nasPathLCFiber + os.sep + m)

        if LCDirFiber.exists() :

            # This is for the mice of fiber photometry

            print('This is a mouse from the LC and fiber photometry data with projection neurons!')

            print('----------Mouse: '+ m + '----------')

            key = {'mouse_name':m}

            ExperimenterNameQuery = 'full_name=' + '"' + (lcProj.Mouse & key).fetch1('responsible_experimenter') + '"'

            experimenter = (lcProj.Person & ExperimenterNameQuery).fetch1('initials')

            print('Loading sessions for mouse: '+ m)

            InfoDir = pathlib.Path(nasPathLCFiber+ m +'/ExperimentInfo.txt')

            Info = {}
            with open(InfoDir) as f:
                for line in f:
                    (key, val) = line.split()
                    Info[key] = val


            sessions = []
            sessionsDate = []

            for key in Info:

                year = '20' + ''.join(key[0:2])
                month = ''.join(key[2:4])
                day = ''.join(key[4:6])
                SessionDate = year + '-' + month + '-' + day

                keyCheck = {'mouse_name':m,
                            'session_date':SessionDate}


                SessionDateFile = year[2:4] +  month + day

                if not((lcProj.Session & keyCheck)):

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

                        dataPathScored = pathlib.Path(nasPathLCFiber+ m +'/' + SessionDateFile  + '*bt.mat')

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

                        dataPathScored = pathlib.Path(nasPathLCFiber+ m +'/' + SessionDateFile + '*bt.mat')

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
