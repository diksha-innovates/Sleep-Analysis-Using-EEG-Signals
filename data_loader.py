"""Finds and loads Sleep-EDF PSG/Hypnogram file pairs.

Sleep-EDF cassette files share their first 6 characters between a subject's
PSG and Hypnogram file (e.g. "SC4001E0-PSG.edf" / "SC4001EC-Hypnogram.edf"),
and differ after that in a session/scorer code, so pairing is done on that
prefix rather than an exact filename match.
"""

import os
from dataclasses import dataclass

import mne

import config


@dataclass
class SubjectRecording:
    subject_id: str
    psg_path: str
    hypnogram_path: str


def find_subject_pairs(raw_dir=config.RAW_DIR):
    """Walks raw_dir and returns a sorted list of SubjectRecording pairs."""
    psg_files = {}
    hyp_files = {}
    for dirpath, _, filenames in os.walk(raw_dir):
        for fname in filenames:
            if not fname.lower().endswith(".edf"):
                continue
            prefix = fname[:6]
            full_path = os.path.join(dirpath, fname)
            if "PSG" in fname:
                psg_files[prefix] = full_path
            elif "Hypnogram" in fname:
                hyp_files[prefix] = full_path

    pairs = [
        SubjectRecording(subject_id=prefix, psg_path=psg_files[prefix], hypnogram_path=hyp_files[prefix])
        for prefix in sorted(psg_files)
        if prefix in hyp_files
    ]
    return pairs


def load_raw_with_annotations(recording: SubjectRecording):
    """Loads a PSG file and attaches its hypnogram as mne Annotations."""
    raw = mne.io.read_raw_edf(recording.psg_path, preload=True, verbose=False)
    annotations = mne.read_annotations(recording.hypnogram_path)
    raw.set_annotations(annotations, emit_warning=False)
    return raw
