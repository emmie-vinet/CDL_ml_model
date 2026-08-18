import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

FILES = {
    "applications":     "r01_Application_Data_201213_202425.xlsx",
    "stakeholders":     "r02_Stakeholder_List_201213-202425.xlsx",
    "venture_cohort":   "r03_Venture_Cohort_201213_202425.xlsx",
    "mentor_cohort":    "r04_Mentor_Cohort_201213-202425.xlsx",
    "venture_session":  "r05_Venture_Session_201213-202425.xlsx",
    "mentor_session":   "r06_Mentor_Session_201213-202425.xlsx",
    "venture_updates":  "r07_Venture_Updates_201213_202425.xlsx",
    "sgm_registrations":"r08_SGM_Registrations_201213-202425.xlsx",
    "hands_raised":     "r09_Hands_Raised_201415-202425.xlsx",
    "meeting_notes":    "r10_Meeting_Notes_201213-202324.xlsx",
    "financing":        "r11_Financing_Events_201213-202425.xlsx",
}

def load_table(name):
    path = DATA_DIR / FILES[name]
    return pd.read_excel(path)

def load_all_tables():
    return {name: load_table(name) for name in FILES}