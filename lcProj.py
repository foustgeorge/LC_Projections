# ============================================================
#  LC Projection Neurons In Sleep Data Injection Pipeline
#  ------------------------------------------------------------
#  lcProj schema: A DataJoint schema to manage polysomnography
#  and fiber photometry recordings from the locus coeruleus (LC)
#  in mice.
#
#  Features:
#  - Stores EEG and EMG recordings
#  - Handles DF/F0 fiber photometry data
#  - Links experimental metadata with corresponding analysis matrices
#  - Supports multiple paradigms: baseline, sleep deprivation,
#    sensory-enhanced sleep deprivation, and fear conditioning
#
#  For more information  on DataJoint schemas please look at https://github.com/datajoint
#  Author: Georgios Foustoukos
#  Date: 14.08.2024
# ============================================================

# packages to be imported for the schema
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
    #Experimenter Details to be added to the metadata
    initials : varchar(4)    # initials used for this person in the lab
    ----
    username : varchar(32)   # username of the experimenter
    full_name: varchar(255)  # fullname of the experimenter
    """

@schema
class WindowCalculationDFFStates(dj.Lookup):
    definition = """
    #Time in hours from a 12h recording to keep in order to calculate the DFF during different sleep states
    #(only that part of the 12h recording data is kept for analysis)
    window : int time in hours
    ---
    """

@schema
class Rig(dj.Lookup):
    definition = """
    #Information regarding the experimental rig to be added to the metadata
    rig_room: varchar(24)           # for example a location of the institute
    ---
    rig_description : varchar(1024) # for example sleep recording room or anything else
    """


@schema
class Mouse(dj.Manual):
    definition = """
    #Mouse metadata imported by an excel file together with the schema
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
    session_date : varchar(255) # date of acquisition
    session_time : varchar(255) # time of acquisition
    ---
    project_id : varchar(255)   # identifier of the type ofproject the data are part of
    -> [nullable] Person
    -> [nullable] Rig
    file_name : varchar(255)    # path to the file which contains the session
    """

@schema
class FibSession(dj.Manual):
    definition = """
    #Experimental session for fiber photometry of the LC combined with multiple behavioural paradigms
    ->Mouse
    session_date : varchar(255)                        # date of acquisition
    session_time : varchar(255)                        # time of acquisition
    ---
    session_type : varchar(255)                        # what type of session this photometry session is related to what the animals was exposed to
    area_fib : varchar(255)                            # where the fiber was located in the brain
    gcamp_virus_info = null : varchar(255)             # 1 virus metadata
    retro_virus_info = null : varchar(255)             # 2 virus metadata
    gcamp_injection_coordinates = null : varchar(255)  # 1 injection coordinates metadata
    retro_injection_coordinates = null : varchar(255)  # 2 injection coordinates metadata
    area_fib_implantation = null : varchar(255)        # fiber implantation coordinates metadata
    implantation_fib_coordinates = null : varchar(255) # area implantation metadata
    power_led = null : varchar(255)                    # led power used for the fiber photometry system
    led_wavelength = null :varchar(255)                # led wavelength used for the fiber photometry system
    """

@schema
class FibDFF(dj.Computed):
    definition = """
    #Dff signal for fiber photometry measurements
    ->FibSession
    ---
    dff: longblob                               # demodulated and computed DF/F0 using the the isosbestic signal as an F0 (happening outside of this schema)
    timestamps: longblob                        # timestamps of data acquisition
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            FibDFF_autopopulation,
        )

        FibDFF_autopopulation(self, key)

@schema
class LFPGoodQuality(dj.Computed):
    definition = """
    #Checking the quality of the LFP signals only for mice with have local field potential electrode in a same brain area as the retrograde viral injection
    ->FibSession
    ---
    quality_hip: int                         # 1 if the LFP signal is of good quality and 0 otherwise
    quality_pfc: int                         # 1 if the LFP signal is of good quality and 0 otherwise
    quality_s1: int                          # 1 if the LFP signal is of good quality and 0 otherwise
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            LFPGoodQuality_autopopulation,
        )

        LFPGoodQuality_autopopulation(self, key)

@schema
class LFPSignals(dj.Computed):
    definition = """
    #Loading the LFP signals for different recording session depending on their quality and downsample the data
    ->FibSession
    ---
    lfp_s1: longblob                         # lfp singal for the s1 electrode
    lfp_hip: longblob                        # lfp singal for the hip electrode
    lfp_pfc: longblob                        # lfp singal for the pfc electrode
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            LFPSignals_autopopulation,
        )

        LFPSignals_autopopulation(self, key)

@schema
class SDSession(dj.Manual):
    definition = """
    #Experimental session of normal sleep deprivation without stimuli
    ->Mouse
    session_date : varchar(255) # date of acquisition
    session_time : varchar(255) # time of acquisition
    ---
    session_type : varchar(255) # session type used for later analysis
    sd_duration : varchar(255)  # duration of SD in hours
    """

@schema
class FCSession(dj.Manual):
    definition = """
    #Experimental session for fear conditioning paradigms
    ->Mouse
    session_date : varchar(255)            # date of acquisition
    session_time : varchar(255)            # time of acquisition
    ---
    session_type : varchar(255)            # session type used for later analysis
    stim_number : int                      # number of repetitions of the sound
    footshock_int : int                    # footshock current intensity in mA
    footshock_dur : int                    # footshock current duration in s
    sound_int: int                         # sound intensity in dB
    sound_dur: int                         # sound duration in seconds
    sound_freq: int                        # sound frequency in Hertz
    sound_footshock_overlap: varchar(255)  # seconds of overlap between the sound stimulus and the footshock
    """


@schema
class SSDSession(dj.Manual):
    definition = """
    #Experimental session of sensory enchaned sleep deprivation
    ->Mouse
    session_date : varchar(255) # date of acquisition
    session_time : varchar(255) # time of acquisition
    ---
    session_type : varchar(255) # session type used for later analysis
    ssd_duration : varchar(255) # duration of SSD in hours
    """

@schema
class BaselineSession(dj.Manual):
    definition = """
    #Experimental session of baseline fiber photometry recording combined with sleep
    ->Mouse
    session_date : varchar(255)      # date of acquisition
    session_time : varchar(255)      # time of acquisition
    ---
    session_type : varchar(255)      # session type used for later analysis
    baseline_duration : varchar(255) # duration of the baseline recording session
    """

@schema
class Hypnogram(dj.Computed):
    definition = """
    #Insertion of the hypnograms from the scored files in the DataJoint corresponding Table
    -> Session
    ---
    hypnos_states = null : longblob        # states automatically determined by the closed-loop sleep state detection algorithm
    hypnos_states_to_plot = null :longblob # states transformed for easier hypnogram plotting
    bfile = null : blob                    # w,r,n string containing the scored sleep states
    bfile_to_plot = null : longblob        # w,r,n string transformed for easier hypnogram plotting
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            Hypnogram_autopopulation,
        )

        Hypnogram_autopopulation(self, key)

@schema
class REMLatency(dj.Computed):
    definition = """
    #Calculate the latency to the first consolidated REM episode
    ->Hypnogram
    session_info :  varchar(225)
    ---
    consolidated_rem : int          # REM duration used for detecting the first consolidated REM episode
    rem_latency : int               # latency in sec to the first consolidated REM
    rem_latency_wake_cor : int      # latency in sec to the first consolidated REM corrected for long wake episodes
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            REMLatency_autopopulation,
        )

        REMLatency_autopopulation(self, key)


@schema
class MADensity(dj.Computed):
    definition = """
    #Calculate the density of microarousals in NREM sleep after SD or SSD
    ->Hypnogram
    session_info :varchar(225)   # session information for further categorisation
    ---
    ma_density : float           # density of MA per minute of NREM sleep during the first our of sleep after the SD or SSD
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
             MADensity_autopopulation,
        )

        MADensity_autopopulation(self, key)



@schema
class MADynamics(dj.Computed):
    definition = """
    #Calculation of the timecourse of microarousals
    -> Hypnogram
    ---
    ma_density = null : longblob    # timecourse of microarousals
    timepoints = null :longblob     # timepoints of the calculation of microarousals
    per_ma = null : longblob        # relative percentage of every type of MAs
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            MADynamics_autopopulation,
        )

        MADynamics_autopopulation(self, key)

@schema
class DFFSleepStates(dj.Computed):
    definition = """
    #Mean DFF values of LC fiber photomotry data values
    ->FibSession
    ->WindowCalculatioofnDFFStates
    ---
    mean_rem_zscored = null   :float                            # mean rem values after z-scoring the signal
    mean_nrem_zscored = null  :float                            # mean nrem values after z-scoring the signal
    mean_wake_zscored = null :float                             # mean wake values after z-scoring the signal
    mean_rem = null          :float                             # mean rem values without z-score
    mean_nrem = null         :float                             # mean nrem values without z-score
    mean_wake  = null        :float                             # mean wake values without z-score
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            DFFSleepStates_autopopulation,
        )

        DFFSleepStates_autopopulation(self, key)

@schema
class NremRemMeanDFF(dj.Computed):
    definition = """
    #Mean DFF activity between two REM episodes or during REM episodes (binned)
    ->BaselineSession
    ---
    during_rem  :longblob                             # binned DFF values during REM epochs
    inter_rem   :longblob                             # binned DFF values between two REM episodes (NREM sleep)
    inter_rem_length   :longblob                      # binned DFF values between two REM episodes kept seperately to split by length of preceding REM
    rem_length   :longblob                            # lenght of the preceding REM used for inter_rem_length
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            NremRemMeanDFF_autopopulation,
        )

        NremRemMeanDFF_autopopulation(self, key)

@schema
class NremRemMeanDFFSDD(dj.Computed):
    definition = """
    #Mean DFF activity between two consolidated REM episodes
    -> SSDSession
    ---
    inter_rem_length   :longblob                      # binned DFF values between two REM episodes kept seperately to split by length of preceding REM
    rem_length   :longblob                            # lenght of the preceding REM used for inter_rem_length
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            NremRemMeanDFFSSD_autopopulation,
        )

        NremRemMeanDFFSSD_autopopulation(self, key)



@schema
class LCPeaks(dj.Computed):
    definition = """
    #LC peaks extracted for each baseline session during NREM
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
class LCFreq(dj.Computed):
    definition = """
    #Computing the LC frequency after a specific manipulation SD, SSD and compare it to baseline
    ->Session
    ---
    lc_freq :longblob                                # lc frequency windowed
    window_times :longblob                           # the middle point of each window of lc_freq
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            LCFreq_autopopulation,
        )

        LCFreq_autopopulation(self, key)


@schema
class MGT(dj.Computed):
    definition = """
    #Compute the Gabor Morlet Wavelet Transform from the bipolarized EEG signal
    #for different bands of interest the signal has a sampling rate of 100Hz if
    #the original EEG was at 1000 Hz

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
class MGTHipLFP(dj.Computed):
    definition = """
    #Compute the Gabor Morlet Wavelet Transform from the LFP signal
    #for different bands of interest.

    ->LFPGoodQuality
    ---
    delta_hip  :longblob                                # delta band (1.5-4Hz)
    sigma_hip  :longblob                                # sigma band (10-15Hz)
    gamma_hip  :longblob                                # gamma band (60-80Hz)
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            MGTHipLFP_autopopulation,
        )

        MGTHipLFP_autopopulation(self, key)

@schema
class MGTS1LFP (dj.Computed):
    definition = """
    #Compute the Gabor Morlet Wavelet Transform from the LFP signal
    #for different bands of interest.

    ->LFPGoodQuality
    ---
    delta_s1  :longblob                                 # delta band (1.5-4Hz)
    sigma_s1  :longblob                                 # sigma band (10-15Hz)
    gamma_s1  :longblob                                 # gamma band (60-80Hz)
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            MGTS1LFP_autopopulation,
        )

        MGTS1LFP_autopopulation(self, key)
@schema
class MGTPfcLFP(dj.Computed):
    definition = """
    #Ccompute the Gabor Morlet Wavelet Transform from the LFP signal
    #for different bands of interest.

    ->LFPGoodQuality
    ---
    delta_pfc  :longblob                                # delta band (1.5-4Hz)
    sigma_pfc  :longblob                                # sigma band (10-15Hz)
    gamma_pfc  :longblob                                # gamma band (60-80Hz)    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            MGTPfcLFP_autopopulation,
        )

        MGTPfcLFP_autopopulation(self, key)


@schema
class HR(dj.Computed):
    definition = """
    #Compute the heart rate extracted from the bipolarized EMG signal
    ->BaselineSession
    ---
    heart_rate  :longblob                 # beats per minutes
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            HR_autopopulation,
        )

        HR_autopopulation(self, key)


@schema
class  WindowCalculation(dj.Lookup):
    definition = """
    #Time in sec to calculate sleep states in the case the signal is windowed
    window : int                             # window time in sec
    ---
    """


@schema
class TimeSpentTransitionsDynamics(dj.Computed):
    definition = """
    #Compute the time spent in each sleep state and the transition to each sleep state using the hypnogram
    -> Hypnogram
    -> WindowCalculation
    ---
    time_spent_norem : float                # time spend in NREM as percentage for the whole file
    time_spent_rem: float                   # time spend in REM as percentage for the whole file
    time_spent_awake: float                 # time spend in Wake as percentage for the whole file
    noremepochs : longblob                  # extracted epochs of NREM
    remepochs : longblob                    # extracted epochs of REM
    awakeepochs : longblob                  # extracted epochs of Wake
    time_spent_rem_windowed : longblob      # time spend in NREM as percentage windowed according to window parameters
    time_spent_norem_windowed : longblob    # time spend in REM as percentage windowed according to window parameters
    time_spent_awake_windowed : longblob    # time spend in Wake as percentage windowed according to window parameters
    rem_transitions_windowed : longblob     # transitions to REM sleep
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            TimeSpentTransitionsDynamics_autopopulation,
        )

        TimeSpentTransitionsDynamics_autopopulation(self, key)
@schema
class FCBehaviour(dj.Computed):
    definition = """
    #Insert the fear conditioning data to DataJoint
    -> FCSession
    ---
    freezing_time : longblob                # absolute freezing time per bin as computed by Ethovision
    freezing_per : longblob                 # freezing percentage related to the duration of a bin
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            FCBehaviour_autopopulation,
        )

        FCBehaviour_autopopulation(self, key)

@schema
class  DFFMeanEqualTimesNREM(dj.Computed):
    definition = """
    #Compute the mean DFF LC activity during NREM in equal times spent in NREM
    ->Session
    ---
    dff_mean :longblob                               # mean DFF in NREM
    window_times :longblob                           # the middle point of each window of dff_mean
    """

    def _make_tuples(self, key):
        from Schema.AutopopulationFunctionsLCProj import (
            DFFMeanEqualTimesNREM_autopopulation,
        )

        DFFMeanEqualTimesNREM_autopopulation(self, key)
