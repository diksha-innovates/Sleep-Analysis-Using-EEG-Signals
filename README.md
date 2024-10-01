# Sleep-Analysis-Using-EEG-Signals
## Dataset : Sleep-EDF Database Expanded
### Dataset Information

    - Format used here is 'European Data Format (EDF)', a standard for storing biosignals(extension .edf).
#### File Types: 
    - '*PSG.edf' : Whole-night polysmnographic sleep recordings containing EEG (from Fpz-Cz and Pz-Oz electrode locations), EOG (horizontal), submental chin EMG, and an event marker.
    - 'SC*PSG.edf' : 'Sleep cassette study' may contain oro-nasal respiration and rectal body temperature.
    - '*Hypnogram.edf' : Sleep stages are annotated in 30-second epochs using labels: Wake (W), N1, N2, N3 (NREM sleep stages), and REM.
#### Duration: 
    Each file represents approximately 24 hours of data per participant.

#### Channels:
    - EEG (Fpz-Cz, Pz-Oz)
    - EOG (horizontal eye movements) 
    - EMG (chin muscles)
    - Respiration signals (from sensors on the chest or abdomen)
    - Heart rate (derived from ECG)
#### Files Name:
    File Prefixes:
      - SC: Denotes recordings from healthy subjects (e.g., SC4001E0.edf).
      - ST: Denotes recordings from patients with sleep disorders (e.g., ST7002J0.edf).
    Suffixes:
      - E0 or E1: Indicate recordings from the first or second night of a subject.
      - PSG: Indicates that the file contains PSG signals.
      - Hyp: Indicates that the file contains hypnogram annotations.

### Dataset Preprocessing



### Dataset Citations

[1] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185–1194 (2000)

[2] MS Mourtazaev, B Kemp, AH Zwinderman, HAC Kamphuisen. Age and gender affect different characteristics of slow waves in the sleep EEG. Sleep 18(7):557–564 (1995).

[3] A Rechtschaffen, AE Kales. A manual of standardized terminology, techniques and scoring systems for sleep stages of human subjects. Los Angeles, CA: UCLA Brain Information Service. Brain Research Institute 10 (1968).

[4] B van Sweden, B Kemp, HAC Kamphuisen, EA van der Velde. Alternative electrode placement in (automatic) sleep scoring (Fpz-Cz/Pz-Oz versus C4-A1/C3-A2). Sleep 13(3):279–283 (1990).

[5]B Kemp, A Värri, AC Rosa, KD Nielsen, J Gade. A simple format for exchange of digitized polygraphic recordings. Electroencephalography and Clinical Neurophysiology 82:391–393 (1992).

[6] B Kemp, J Olivan. European data format 'plus' (EDF+), an EDF alike standard format for the exchange of physiological data. Clinical Neurophysiology 114:1755–1761 (2002).

[7] B Kemp. Model-based monitoring of human sleep stages. Ph.D. thesis, Twente University (1987).

[8] B Kemp, AJMW Janssen, MJ Roessen. A digital telemetry system for ambulatory sleep recording. In: Sleep-Wake Research in The Netherlands 4, eds ML Coenen and J Arends, Dutch Society for Sleep Wake Research, pp 129–132 (1993).
