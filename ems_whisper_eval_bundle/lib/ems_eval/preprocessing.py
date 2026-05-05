"""Preprocessing utilities for EMS text normalization."""

import re
from typing import Dict

EMS_ABBREVIATIONS: Dict[str, str] = {
    # Vital signs & patient status
    "pt": "patient",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "spo2": "SpO2",
    "etco2": "EtCO2",
    "gcs": "GCS",
    "avpu": "AVPU",
    "ros": "rule out",
    "aox": "alert and oriented",
    "aox3": "alert and oriented times three",
    "aox4": "alert and oriented times four",
    # Routes & administration
    "iv": "intravenous",
    "io": "intraosseous",
    "im": "intramuscular",
    "po": "by mouth",
    "sl": "sublingual",
    "in": "intranasal",
    "sq": "subcutaneous",
    # Equipment & procedures
    "bvm": "bag valve mask",
    "cpr": "cardiopulmonary resuscitation",
    "aed": "automated external defibrillator",
    "defib": "defibrillation",
    "et": "endotracheal",
    "ett": "endotracheal tube",
    "cric": "cricothyrotomy",
    "ng": "nasogastric",
    "og": "orogastric",
    "cpap": "continuous positive airway pressure",
    "bipap": "bilevel positive airway pressure",
    "rsi": "rapid sequence intubation",
    # Cardiac & arrest
    "rosc": "return of spontaneous circulation",
    "pea": "pulseless electrical activity",
    "vf": "ventricular fibrillation",
    "vt": "ventricular tachycardia",
    "stemi": "ST elevation myocardial infarction",
    "nstemi": "non-ST elevation myocardial infarction",
    "acs": "acute coronary syndrome",
    "mi": "myocardial infarction",
    # Trauma & incidents
    "mvc": "motor vehicle collision",
    "mva": "motor vehicle accident",
    "gsw": "gunshot wound",
    "stabbing": "stab wound",
    "sw": "stab wound",
    "fx": "fracture",
    "lac": "laceration",
    "contusion": "contusion",
    # Drugs (common abbreviations)
    "epi": "epinephrine",
    "lido": "lidocaine",
    "mag": "magnesium",
    "bicarb": "bicarbonate",
    "ns": "normal saline",
    "lr": "lactated ringers",
    "d5w": "dextrose 5% in water",
    "d50": "dextrose 50%",
    "txa": "tranexamic acid",
    "asa": "aspirin",
    "nitro": "nitroglycerin",
    "adeno": "adenosine",
    "amio": "amiodarone",
    "narcan": "naloxone",
    "versed": "midazolam",
    "ketamine": "ketamine",
    "fentanyl": "fentanyl",
    "morphine": "morphine",
    "benadryl": "diphenhydramine",
    "zofran": "ondansetron",
    "reglan": "metoclopramide",
    "glucagon": "glucagon",
    "insulin": "insulin",
    "albuterol": "albuterol",
    "atrovent": "ipratropium",
    # Assessment mnemonics
    "sample": "SAMPLE",
    "opqrst": "OPQRST",
    "dcap": "DCAP-BTLS",
    "btls": "DCAP-BTLS",
    # Units & levels
    "systolic": "systolic",
    "diastolic": "diastolic",
    "mg": "milligrams",
    "mcg": "micrograms",
    "ml": "milliliters",
    "units": "units",
    "cc": "cubic centimeters",
    # Other common EMS
    "als": "advanced life support",
    "bls": "basic life support",
    "eta": "estimated time of arrival",
    "pd": "police department",
    "ed": "emergency department",
    "er": "emergency room",
    "hospital": "hospital",
    "copd": "COPD",
    "chf": "congestive heart failure",
    "dm": "diabetes mellitus",
    "htn": "hypertension",
    "cva": "cerebrovascular accident",
    "tia": "transient ischemic attack",
    "od": "overdose",
    "etoh": "ethanol",
    "dka": "diabetic ketoacidosis",
    "hypoglycemia": "hypoglycemia",
    "hyperglycemia": "hyperglycemia",
    "sepsis": "sepsis",
    "anaphylaxis": "anaphylaxis",
    "intubation": "intubation",
    "decompression": "needle decompression",
}


def normalize_ems_text(text: str) -> str:
    """
    Normalize EMS text for evaluation: expand abbreviations, lowercase,
    remove punctuation (keep hyphens in medical terms), normalize whitespace.

    Args:
        text: Raw EMS transcript or hypothesis text.

    Returns:
        Normalized text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Lowercase for consistent matching
    result = text.lower().strip()

    # Expand abbreviations (match whole words, case-insensitive)
    words = result.split()
    expanded: list[str] = []
    for w in words:
        # Remove trailing punctuation for lookup (e.g., "pt," -> "pt")
        clean = re.sub(r"^([\w\-]+).*", r"\1", w)
        if clean in EMS_ABBREVIATIONS:
            expanded.append(EMS_ABBREVIATIONS[clean])
        else:
            expanded.append(w)

    result = " ".join(expanded)

    # Remove punctuation except hyphens within words (medical terms like "non-ST-elevation")
    # Keep hyphens that are part of medical terms (e.g., "pre-hospital", "self-reported")
    result = re.sub(r"[^\w\s\-]", " ", result)
    # Collapse multiple hyphens or spaces around hyphens
    result = re.sub(r"\s*-\s*", "-", result)

    # Normalize whitespace: collapse multiple spaces, strip
    result = re.sub(r"\s+", " ", result).strip()

    return result
