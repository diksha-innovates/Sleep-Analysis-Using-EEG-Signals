"""Trains and evaluates a real sleep-stage classifier on band-power features.

Replaces the original notebook's cells 8-9, which computed metrics on a
hardcoded 10-sample example array instead of any actual prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

import config


def train_sleep_stage_classifier(X, y, groups=None, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """StandardScaler + RandomForest, with a train/test split chosen by `groups`.

    `groups` should be one subject id per row of X/y. Epochs from the same
    subject are correlated (same brain, same electrodes, same night), so a
    plain random/stratified split lets epochs from one subject land in both
    train and test -- the model partly memorizes that subject rather than
    generalizing, which inflates every reported metric. When `groups` has
    at least 2 distinct subjects, GroupShuffleSplit keeps each subject
    entirely on one side of the split. With a single subject (or no groups
    given), there's nothing to group by, so it falls back to a stratified
    per-epoch split and the caller should treat the resulting metrics as
    optimistic.

    Returns (model, scaler, X_test_scaled, y_test, y_pred, groups_test), where
    groups_test is the subject id of each test row (None if `groups` wasn't
    given) -- callers can use it to confirm train/test subjects are disjoint.
    """
    groups_test = None
    if groups is not None and len(set(groups)) > 1:
        groups_arr = np.asarray(groups)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups_arr))
        X_train, X_test = np.asarray(X)[train_idx], np.asarray(X)[test_idx]
        y_train, y_test = np.asarray(y)[train_idx], np.asarray(y)[test_idx]
        groups_test = groups_arr[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    return model, scaler, X_test_scaled, y_test, y_pred, groups_test
