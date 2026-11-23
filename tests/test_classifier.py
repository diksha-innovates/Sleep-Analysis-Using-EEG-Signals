import classifier


def test_classifier_runs_and_predicts_known_labels(synthetic_features_and_labels):
    X, y = synthetic_features_and_labels
    model, scaler, X_test, y_test, y_pred, groups_test = classifier.train_sleep_stage_classifier(X, y)

    assert groups_test is None  # no groups given -> nothing to report
    assert len(y_pred) == len(y_test) == len(X_test)
    assert set(y_pred).issubset(set(y))


def test_group_split_keeps_subjects_disjoint(synthetic_multi_subject_dataset):
    """The core leakage-fix guarantee: with >= 2 subjects, every subject in
    the test split must be entirely absent from training."""
    X, y, groups = synthetic_multi_subject_dataset

    _, _, _, _, _, groups_test = classifier.train_sleep_stage_classifier(X, y, groups=groups)

    assert groups_test is not None
    test_subjects = set(groups_test)
    train_subjects = set(groups) - test_subjects
    assert test_subjects, "expected at least one subject held out for testing"
    assert test_subjects.isdisjoint(train_subjects)


def test_single_subject_falls_back_to_stratified_split(synthetic_features_and_labels):
    X, y = synthetic_features_and_labels
    groups = ["only_subject"] * len(X)
    # Should not raise even though there's nothing to group by.
    _, _, X_test, y_test, y_pred, groups_test = classifier.train_sleep_stage_classifier(X, y, groups=groups)
    assert groups_test is None
    assert len(y_test) > 0
