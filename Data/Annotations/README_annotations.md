## README -- CASE_dataset/data/raw/annotations

This short guide to the raw annotation data, covers the following topics:

1. General Information.


### 1. General Information:
This folder contains the raw annotation data for all 30 subjects.
Each file (e.g., sub1_joystick.txt) contains the following 3 variables
(1 variable per column):

1. jstime: is the time provided by LabVIEW while logging. It is the global time
and is also used for physiological files to allow synchronization of data across
the two files. It is named jstime to keep the variable name different from 
daqtime (used for physiological data). Measurements in seconds.

2. X axis: is the valence data in the integer range from -26225 to +26225.

3. Y axis: is the arousal data in the integer range from -26225 to +26225.


