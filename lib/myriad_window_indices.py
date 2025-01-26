"""
This file presents the indices of the short-time windows that are used to
isolate the reflections in the matched RIRs. The window size is set to 96 samples
and the reflections are expected to be located around the middle of the window.
"""

import numpy as np

win_size = 96
refl_start_indices = np.array(
    [
        [298, 450, 1040, 1210],
        [215, 390, 1020, 1190],
        [349, 475, 1085, 1210],
        [540, 650, 1150, 1270],
        [644, 740, 1050, 1180],
        [490, 590, 950, 1095],
        [400, 515, 885, 1080],
        [450, 560, 1030, 1275],
        [425, 650, 1130, 1260],
        [355, 610, 850, 1150],
        [450, 660, 1205, 1860],
        [620, 790, 1290, 1690],
        [575, 765, 990, 1200],
        [610, 790, 925, 1310],
        [720, 870, 1280, 1440],
        [560, 750, 870, 1200],
        [490, 690, 1050, 1160],
        [545, 735, 1080, 1320],
        [420, 650, 810, 900],
        [370, 610, 1060, 1160],
    ]
)
