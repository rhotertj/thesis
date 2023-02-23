import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def generate_class_description_fine(row):
    pass_technique = {
        "O":  None,
        "A": "Schlagwurfpass",
        "B": "Handgelenkspass",
        "C": "Druckpass",
        "D": "Rückhandpass",
        "E": "Undefinierter Pass",
        "X":  None
    }
    shot_technique = {
        0: None,
        1: "Sprungwurf Außen",
        2: "Sprungwurf Rückraum",
        3: "Sprungwurf Torraum/Zentrum",
        4: "Standwurf mit Anlauf",
        5: "Standwurf ohne Anlauf",
        6: "Drehwurf",
        7: "7-Meter",
        8: "Undefinierter Wurf", 
    }
    if row["shot"] == None:
        return "Background"
    elif row["shot"] == "0" and row["pass"] in ("O", "X"):
        return "Foul"
    elif row["pass"] in ("O", "X"):
        return shot_technique[int(row["shot"])]
    elif row["shot"] == "0":
        return pass_technique[row["pass"]]
    else:
        print("error at", row["pass"], row["shot"], row["label"], type(row["pass"]), type(row["shot"]))

def generate_class_description_coarse(fine_class):
    pass_technique = {
    "O":  None,
    "A": "Schlagwurfpass",
    "B": "Handgelenkspass",
    "C": "Druckpass",
    "D": "Rückhandpass",
    "E": "Undefinierter Pass",
    "X":  None
    }
    shot_technique = {
        0: None, # zero
        1: "Sprungwurf Außen",
        2: "Sprungwurf Rückraum",
        3: "Sprungwurf Torraum/Zentrum",
        4: "Standwurf mit Anlauf",
        5: "Standwurf ohne Anlauf",
        6: "Drehwurf",
        7: "7-Meter",
        8: "Undefinierter Wurf", 
    }
    if fine_class in pass_technique.values():
        return "Pass"
    if fine_class in shot_technique.values():
        return "Shot"
    else:
        return fine_class # Background and Foul


