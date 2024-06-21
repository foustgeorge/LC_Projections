import datajoint as dj
import os, sys
import csv
import pathlib
import numpy as np
from tqdm import tqdm
from tqdm import trange
import scipy.io as sio
import datetime


schema = dj.schema(
    "lc_proj_schema"
)

@schema
class Person(dj.Lookup):
    definition = """
    #Experimenter Details
    initials : varchar(4) #initials used for this person in the lab
    ----
    username : varchar(32) #username of the experimenter
    full_name: varchar(255)
    """

@schema
class WindowCalculationDFFStates(dj.Lookup):
    definition = """
    #Time in hours from a 12h recording to keep in order to calculate the DFF during different sleep states
    window : int
    ---
    """

@schema
class Rig(dj.Lookup):
    definition = """
    rig_room: varchar(24) #for example Neurobau2
    ---
    rig_description : varchar(1024) #for example miniscope room or anything else
    """


@schema
class Mouse(dj.Manual):
    definition = """
    #Mouse details as being imported by slims
    mouse_name: varchar(32)                      # unique mouse name
    ---
    responsible_experimenter: varchar(32)        # person responsible for this mouse
    pyrat_id: varchar(32)                        # mouse id as in Pyrat
    genotype: varchar(32)                        # mouse strain as in Pyrat
    gender : varchar(32)                         # mouse gender as in Pyrat
    birth_date: varchar(24)                      # date of birth of the mouse as in Pyrat
    licence: varchar(12)                         # licence number of the experiment
    dg: float                                    # degree of severity according to animal licence
    start_date: varchar(32)                      # start date of the experiment
    age: varchar(32)                             # age of the animal the time of the experiment
    sacrifice_date: varchar(32)                  # sacrifice date of the animal
    filename: varchar(255)                       # the name of the file from which the data were injected
    proj_type : varchar(255)                     # LC projection neuron type measured for this mouse
    """

@schema
class Session(dj.Manual):
    definition = """
    #Experimental session for a specific mouse and a project
    ->Mouse
    session_date : varchar(255)
    session_time : varchar(255)
    ---
    project_id : varchar(255)
    -> [nullable] Person
    -> [nullable] Rig
    file_name : varchar(255)
    """

@schema
class FibSession(dj.Manual):
    definition = """
    #Experimental session for fiber photometry of the LC with or without SD/SSD
    ->Mouse
    session_date : varchar(255)
    session_time : varchar(255)
    ---
    session_type : varchar(255)
    area_fib : varchar(255)
    gcamp_virus_info = null : varchar(255)
    retro_virus_info = null : varchar(255)
    gcamp_injection_coordinates = null : varchar(255)
    retro_injection_coordinates = null : varchar(255)
    area_fib_implantation = null : varchar(255)
    implantation_fib_coordinates = null : varchar(255)
    power_led = null : varchar(255)
    led_wavelength = null :varchar(255)
    """

@schema
class FibDFF(dj.Computed):
    definition = """
    #Dff signal for fiber photometry measurements 
    ->FibSession
    ---
    dff: longblob                               # demodulated and computed DF/F0 using the the isosbestic signal as an F0
    timestamps: longblob                        # timestamps of data acquisition
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            FibDFF_autopopulation,
        )

        FibDFF_autopopulation(self, key)



@schema
class SDSession(dj.Manual):
    definition = """
    #Experimental session of normal sleep deprivation without stimuli
    ->Mouse
    session_date : varchar(255)
    session_time : varchar(255)
    ---
    session_type : varchar(255)
    sd_duration : varchar(255)
    """

@schema
class SSDSession(dj.Manual):
    definition = """
    #Experimental session of sensory enchaned sleep deprivation
    ->Mouse
    session_date : varchar(255)
    session_time : varchar(255)
    ---
    session_type : varchar(255)
    ssd_duration : varchar(255)
    """

@schema
class BaselineSession(dj.Manual):
    definition = """
    #Experimental of baseline fiber photometry
    ->Mouse
    session_date : varchar(255)
    session_time : varchar(255)
    ---
    session_type : varchar(255)
    baseline_duration : varchar(255)
    """

@schema
class Hypnogram(dj.Computed):
    definition = """
    #  Insertion of the hypnograms in the datajoint from the scored files
    -> Session
    ---
    hypnos_states = null : longblob
    hypnos_states_to_plot = null :longblob
    bfile = null : blob
    bfile_to_plot = null : longblob
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            Hypnogram_autopopulation,
        )

        Hypnogram_autopopulation(self, key)

@schema
class REMLatency(dj.Computed):
    definition = """
    # Calculate the latency to the first consolidated rem
    ->Hypnogram
    session_info :  varchar(225)
    ---
    consolidated_rem : int # number used for the first consolidated rem
    rem_latency : int # latency in sec to the first consolidated rem
    rem_latency_wake_cor : int # latency in sec to the first consolidated rem corrected for long wake episodes
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            REMLatency_autopopulation,
        )

        REMLatency_autopopulation(self, key)


@schema
class MADensity(dj.Computed):
    definition = """
    # Calculate the latency to the first consolidated rem
    ->Hypnogram
    session_info :  varchar(225)
    ---
    ma_density : float # density of MA per minute of NREM sleep during the first our of sleep after the SD or SSD
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
             MADensity_autopopulation,
        )

        MADensity_autopopulation(self, key)



@schema
class MADynamics(dj.Computed):
    definition = """
    #  Calculation of the timecourse of microarousals
    -> Hypnogram
    ---
    ma_density = null : longblob # timecourse of microarousals
    timepoints = null :longblob # timepoints of the calculation of microarousals
    per_ma = null : longblob # relative percentage of every type of MAs
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            MADynamics_autopopulation,
        )

        MADynamics_autopopulation(self, key)




@schema
class LCActivitySDvsSSD(dj.Computed):
    definition = """
    ->Mouse
    session_info :  varchar(225)
    ---
    lc_activity = null : longblob
    times = null : longblob
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            LCActivitySDvsSSD_autopopulation,
        )

        LCActivitySDvsSSD_autopopulation(self, key)




class SSDREMBoutsOpto(dj.Computed):

    definition = """
    #Experimental session of sensory enchaned sleep deprivation
    ->SSDSession
    ---
    session_type : varchar(255)
    time_spent_rem = null : float
    transitions_rem = null : float
    rem_bouts = null : float
    """
    def _make_tuples(self, key):

        from Schema.AutopopulationFunctionsLCProj import (
            SSDREMBoutsOpto_autopopulation,)

        SSDREMBoutsOpto_autopopulation(self, key)

@schema
class REMInfoSDvsSSD(dj.Computed):

    definition = """
    # Information of REM during the first hour after SD and SSD session
    ->Hypnogram
    session_info :  varchar(225)
    ---
    time_spent_rem = null : float
    transitions_rem = null : float
    rem_bouts = null : float
    """
    def _make_tuples(self, key):

        from Schema.AutopopulationFunctionsLCProj import (
            REMInfoSDvsSSD_autopopulation,)

        REMInfoSDvsSSD_autopopulation(self, key)

@schema
class DFFSleepStates(dj.Computed):
    definition = """
    # Mean DFF values for each sleep state 
    ->BaselineSession
    ->WindowCalculationDFFStates
    ---
    mean_rem_zscored   :float                             # mean rem values after z-scoring the signal 
    mean_nrem_zscored  :float                             # mean nrem values after z-scoring the signal 
    mean_wake_zscored  :float                             # mean wake values after z-scoring the signal 
    mean_rem           :float                             # mean rem values without z-score
    mean_nrem          :float                             # mean nrem values without z-score
    mean_wake          :float                             # mean wake values without z-score
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            DFFSleepStates_autopopulation,
        )

        DFFSleepStates_autopopulation(self, key)

@schema
class NremRemMeanDFF(dj.Computed):
    definition = """
    # Mean DFF activity between two REM episodes or during REM episodes (binned)
    ->BaselineSession
    ---
    during_rem  :longblob                             # binned DFF values during REM epochs
    inter_rem   :longblob                             # binned DFF values between two REM episodes (NREM sleep)
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            NremRemMeanDFF_autopopulation,
        )

        NremRemMeanDFF_autopopulation(self, key)


@schema
class LCPeaks(dj.Computed):
    definition = """
    # LC peaks extracted for each baseline session during NREM 
    ->BaselineSession
    ---
    lc_peaks  :longblob                                # position of the peaks (indexes)
    peaks_flag   :longblob                             # flag 0 peak with no MA or 1 peak with MA
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            LCPeaks_autopopulation,
        )

        LCPeaks_autopopulation(self, key)

@schema
class MGT(dj.Computed):
    definition = """
    # compute the Gabor Morlet Wavelet Transform from the bipolarized EEG signal
    # for different bands of interest the signal has a sampling rate of 100Hz if
    # the original EEG was at 1000 Hz
    
    ->BaselineSession
    ---
    delta  :longblob                                # delta band (1.5-4Hz)
    sigma  :longblob                                # sigma band (10-15Hz)
    gamma  :longblob                                # gamma band (60-80Hz)
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            MGT_autopopulation,
        )

        MGT_autopopulation(self, key)


@schema
class HR(dj.Computed):
    definition = """
    # compute the heart rate extracted from the bipolarized EMG signal
    ->BaselineSession
    ---
    heart_rate  :longblob                                # beats per minutes 
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            HR_autopopulation,
        )

        HR_autopopulation(self, key)





