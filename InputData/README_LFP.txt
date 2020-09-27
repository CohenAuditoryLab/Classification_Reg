LFP data for ABBA streaming task

param
basic information of the session

LFP 
neural signal sampled at 1017.252625 Hz.
trial x time (sample) x channel

t
time correspond for the RAW data.
0 means the onset of first tone A.

targetTime
location of the target in each trial (either 675, 900, 1125 or 1350 ms).

stDiff
semitone difference between tone A and tone B.

index
type of monkey's behavior
    0 - hit
    1 - miss
    2 - false alarm
    3 - start error (should ignore for the analysis)
    4 - touch error (should ignore for the analysis)

index_tsRT
trial numbers for a special case of false alarm trials in which target was presented but monkey respond too short response latency (<200 ms).

Stim1
copy of the stimulus presented in each trial.
sampling rate = 1017.252625 Hz

Lever
lever status
sampling rate = 1017.252625 Hz

Rew
timing of the reward
sampling rate = 1017.252625 Hz
