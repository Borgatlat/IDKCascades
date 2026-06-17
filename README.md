Trying to recreate the results of the Timely Classification of Hierarchical Classes (https://eprints.whiterose.ac.uk/id/eprint/231704/) paper

This repo will focus on the M3N-VC dataset

The dataset (used in the paper) comprises of
 - 4 base vehicle types(classes) [MUSTANG, MX5, CX30, GLE350]
 - 3.43h (h24)
 - 2 second windows
 - turned into spectograms

Classifiers (used in the paper)
 - 3 levels of DeepSense Classifiers
    - Globals [MUSTANG, MX5, CX30, GLE350, BACKGROUND]
    - Intermediate [SUV, COUPE, BACKGROUND]
    - Specialized [SUV, COUPE]

Classifier Table
+-----------+-----------+-----------+-----------+
|Classifiers|Type       |Params     |Data       |
+-----------+-----------+-----------+-----------+
|K0         |Inter      |129698     |Both       |
+-----------+-----------+-----------+-----------+
|K1         |Inter      |356610     |Both       |
+-----------+-----------+-----------+-----------+
|K2         |Global     |130469     |Both       |
+-----------+-----------+-----------+-----------+
|K3         |Global     |1217109    |Both       |
+-----------+-----------+-----------+-----------+
|K4         |Spec(SUV)  |80355      |Accoustic  |
+-----------+-----------+-----------+-----------+
|K5         |Spec(Coupe)|80355      |Accoustic  |
+-----------+-----------+-----------+-----------+
|K6         |Spec(Coupe)|129955     |Both       |
+-----------+-----------+-----------+-----------+