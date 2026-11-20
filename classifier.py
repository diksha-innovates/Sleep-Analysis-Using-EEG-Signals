"""Trains and evaluates a real sleep-stage classifier on band-power features.

Replaces the original notebook's cells 8-9, which computed metrics on a
hardcoded 10-sample example array instead of any actual prediction.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def train_sleep_stage_classifier(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """Stratified train/test split + StandardScaler + RandomForest.

    Returns (model, scaler, X_test_scaled, y_test, y_pred).
    """
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

    return model, scaler, X_test_scaled, y_test, y_pred
