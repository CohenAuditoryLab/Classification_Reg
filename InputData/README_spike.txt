1Spike data for ABBA streaming task

clInfo
basic information of the cluster spike sorted by Kilosort
 - all_cluster 
   cluster ID of all sorted units
 - channel
   recorded channel of all sorted units
 - active_cluster
   cluster ID of responding units
 - active_channel
   recorded channel of responding units
 - pvalue
   p-value comparing spiking activity between spontaneous and stimulus period
 - sorting_quality_ks
   sorting quality manually assigned for each unit
   0 - noise, 1 - MUA, 2 - Good, 3- Unsorted, 4 - Drift
 - sorting_quality_isi
   sorting quality based on ISI
   0 - MUA, 1 - Single
 - isi_violation_value
   p-value for ISI violation (coming from Matt's code)

trialInfo
informataion for each trial
 - target_time
   location of the target in each trial (either 675, 900, 1125 or 1350 ms)
 - semitone_diff
   semitone difference between tone A and tone B
 - behav_ind
   type of monkey's behavior
    0 - hit
    1 - miss
    2 - false alarm
    3 - start error (should ignore for the analysis)
    4 - touch error (should ignore for the analysis)
 - behav_ind_ts
   trial numbers for a special case of false alarm trials in which target was presented but monkey respond too short response latency (<200 ms).

SpikeTiming
time stamp data
   1st column - spike timing in ms
   2nd column - cluster ID

Raster
binned spike data (1 ms bin).
trial x time (sample) x unit

t_raster
time correspond for the Raster data.
0 means the onset of first tone A.

leverRelease
lever release timing for each trial

Stim1
copy of the stimulus presented in each trial.
sampling rate = 1017.252625 Hz

Lever
lever status
sampling rate = 1017.252625 Hz

Rew
timing of the reward
sampling rate = 1017.252625 Hz

