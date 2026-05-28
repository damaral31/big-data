# Data schemas and structures for MIMIC-III
from typing import NamedTuple, List, Optional
from datetime import datetime

class Patient(NamedTuple):
    """Patient information"""
    subject_id: int
    gender: str
    dob: Optional[datetime]

class Admission(NamedTuple):
    """Hospital admission information"""
    hadm_id: int
    subject_id: int
    admittime: datetime
    dischtime: datetime
    admission_type: str
    length_of_stay_actual: float  # Target variable

class ICUStay(NamedTuple):
    """ICU stay information"""
    icustay_id: int
    hadm_id: int
    subject_id: int
    intime: datetime
    outtime: datetime
    los_icu: float

class ChartEvent(NamedTuple):
    """Chart event record"""
    icustay_id: int
    hadm_id: int
    subject_id: int
    charttime: datetime
    itemid: int
    value: float
    valueuom: str
    label: str

class Item(NamedTuple):
    """Item (measurement) information"""
    itemid: int
    label: str
    abbreviation: str
    category: str

class SampleData(NamedTuple):
    """Training sample with features and target"""
    patient_id: int
    admission_id: int
    features: dict  # Feature name -> value
    target: float   # Length of stay in days
    sample_time: datetime  # Time of prediction window
    time_until_discharge: float  # Hours remaining until discharge
