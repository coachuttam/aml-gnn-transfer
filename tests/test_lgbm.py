"""Tests for the Phase E LightGBM baseline (small made-up data, no real LI-Small)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src import config, lgbm_model, lgbm_run
from src.label_mask import fingerprint
from src.lgbm_data import DaySplit, day_split, load_in_multignn_order
from src.lgbm_features import FEATURE_COLUMNS, add_account_codes
from src.lgbm_inputs import LgbmInputs, build_inputs
from src.metrics import best_f1_threshold

RAW_HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
    "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
)


def raw_row(minute: int, sender: str, receiver: str, label: int = 0) -> str:
    day, minute_of_day = divmod(minute, 24 * 60)
    stamp = "2022/09/%02d %02d:%02d" % (
        1 + day,
        minute_of_day // 60,
        minute_of_day % 60,
    )
    return "%s,10,%s,20,%s,100.0,US Dollar,100.0,US Dollar,Cheque,%d\n" % (
        stamp, sender, receiver, label,
    )  # fmt: skip


def write_raw(path: Path, rows: List[str]) -> Path:
    path.write_text(RAW_HEADER + "".join(rows))
    return path


# --- split and row order ---


def test_ten_equal_days_split_six_two_two() -> None:
    seconds = np.repeat(np.arange(10) * 86400, 10)  # 10 days, 10 transactions each
    split = day_split(seconds)
    assert (split.train_end, split.validation_end, split.total) == (60, 80, 100)


def test_the_split_cuts_between_whole_days_never_inside_a_day() -> None:
    counts = [30, 5, 5, 20, 20, 10, 5, 5]
    seconds = np.concatenate([np.full(c, d * 86400 + 5) for d, c in enumerate(counts)])
    split = day_split(seconds)
    assert split.train_end in np.cumsum([0] + counts)
    assert split.validation_end in np.cumsum([0] + counts)


def test_the_period_slices_cover_every_row_once() -> None:
    split = DaySplit(6, 8, 10)
    covered = list(range(10))
    assert covered[split.train] + covered[split.validation] + covered[split.test] == (
        covered
    )


def test_rows_come_back_in_time_order_with_ties_in_file_order(tmp_path: Path) -> None:
    path = write_raw(
        tmp_path / "raw.csv",
        [raw_row(30, "a", "b"), raw_row(10, "c", "d"), raw_row(30, "e", "f")],
    )
    frame = load_in_multignn_order(path)
    assert list(frame["from_account"]) == ["c", "a", "e"]
    assert frame["seconds"].tolist() == [0, 1200, 1200]  # smallest value becomes 0


# --- labels never reach the features ---


def test_no_feature_is_named_after_a_label() -> None:
    assert not [name for name in FEATURE_COLUMNS if "laundering" in name]


def test_features_do_not_change_when_labels_change(tmp_path: Path) -> None:
    rows = [raw_row(m, "a" if m % 2 else "b", "c") for m in range(0, 6 * 1440, 300)]
    flipped = [row[:-2] + "1\n" for row in rows]
    first = build_inputs(write_raw(tmp_path / "one.csv", rows))
    second = build_inputs(write_raw(tmp_path / "two.csv", flipped))
    assert np.array_equal(first.features, second.features)
    assert first.truth.sum() == 0 and second.truth.sum() == len(rows)


def test_account_codes_number_each_bank_and_account_pair_once() -> None:
    frame = pd.DataFrame(
        {
            "from_bank": ["1", "2", "1"], "from_account": ["a", "b", "a"],
            "to_bank": ["2", "1", "2"], "to_account": ["b", "a", "z"],
        }
    )  # fmt: skip
    for column in frame:
        frame[column] = frame[column].astype("category")
    coded = add_account_codes(frame)
    assert coded["from_code"].tolist() == [0, 1, 0]  # (1,a) (2,b) (1,a)
    assert coded["to_code"].tolist() == [1, 0, 2]  # (2,b) (1,a) (2,z)


# --- same labels as Phase C ---


def write_phase_c_record(results: Path, fraction: float, seed: int, kept: Any) -> None:
    folder = results / "phases" / "phaseC" / "records"
    folder.mkdir(parents=True, exist_ok=True)
    name = "gin_scratch_f%s_s%d.json" % (float(fraction), seed)
    record = {"config": {"kept_labels_fingerprint": fingerprint(kept)}}
    (folder / name).write_text(json.dumps(record))


def test_kept_labels_that_match_phase_c_pass(tmp_path: Path) -> None:
    kept = lgbm_run.kept_training_rows(1000, 0.25, 1)
    write_phase_c_record(tmp_path, 0.25, 1, kept)
    assert lgbm_run.assert_same_labels_as_phase_c(kept, 0.25, 1, tmp_path)


def test_kept_labels_that_differ_from_phase_c_stop_the_run(tmp_path: Path) -> None:
    write_phase_c_record(tmp_path, 0.25, 1, lgbm_run.kept_training_rows(999, 0.25, 1))
    kept = lgbm_run.kept_training_rows(1000, 0.25, 1)
    with pytest.raises(RuntimeError, match="differ from Phase C"):
        lgbm_run.assert_same_labels_as_phase_c(kept, 0.25, 1, tmp_path)


def test_a_missing_phase_c_record_stops_the_run_too(tmp_path: Path) -> None:
    kept = lgbm_run.kept_training_rows(1000, 0.1, 0)
    with pytest.raises(RuntimeError, match="No Phase C record"):
        lgbm_run.assert_same_labels_as_phase_c(kept, 0.1, 0, tmp_path)


def test_ten_percent_of_the_labels_sit_inside_the_twenty_five_percent_set() -> None:
    small = set(lgbm_run.kept_training_rows(5000, 0.1, 0))
    large = set(lgbm_run.kept_training_rows(5000, 0.25, 0))
    assert small <= large and max(small) < 5000


# --- one run, end to end, on made-up data ---


def made_up_inputs(rows: int = 3000) -> LgbmInputs:
    generator = np.random.RandomState(0)
    features = generator.normal(size=(rows, len(FEATURE_COLUMNS))).astype(np.float32)
    truth = (features[:, 0] + generator.normal(size=rows) > 2.0).astype(np.int8)
    split = DaySplit(int(rows * 0.6), int(rows * 0.8), rows)
    return LgbmInputs(features, truth, split, list(FEATURE_COLUMNS))


@pytest.fixture
def small_model(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(config, "LIGHTGBM_MAX_TREES", 20)
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "data")
    return made_up_inputs()


def run_made_up(inputs: LgbmInputs, tmp_path: Path) -> Dict[str, Any]:
    kept = lgbm_run.kept_training_rows(inputs.split.train_end, 0.5, 0)
    write_phase_c_record(tmp_path, 0.5, 0, kept)
    path = lgbm_run.run_one(inputs, 0.5, 0, tmp_path / "records", tmp_path)
    return json.loads(path.read_text())


def test_a_run_writes_one_record_named_for_its_settings(
    small_model, tmp_path: Path
) -> None:
    record = run_made_up(small_model, tmp_path)
    assert record["run_id"] == "lightgbm_scratch_f0.5_s0"
    assert record["model"] == "lightgbm" and record["mode"] == "scratch"
    assert record["config"]["status"] == "completed"


def test_the_record_has_the_same_fields_as_a_phase_c_record(
    small_model, tmp_path: Path
) -> None:
    record = run_made_up(small_model, tmp_path)
    reference_path = next(
        (config.PHASES_RESULTS_DIR / "phaseC" / "records").glob("*.json")
    )
    reference = json.loads(reference_path.read_text())
    assert set(record) == set(reference)
    assert set(record["metrics"]["scored"]) == set(reference["metrics"]["scored"])
    assert set(reference["config"]) <= set(record["config"])


def test_a_finished_run_is_skipped_not_overwritten(small_model, tmp_path: Path) -> None:
    run_made_up(small_model, tmp_path)
    again = lgbm_run.run_one(small_model, 0.5, 0, tmp_path / "records", tmp_path)
    assert again is None


def test_training_sees_only_the_kept_labelled_training_rows(
    small_model, tmp_path: Path, monkeypatch
) -> None:
    seen: List[np.ndarray] = []
    original = lgbm_model.fit_one_weight

    def spy(inputs, rows, *others):
        seen.append(rows)
        return original(inputs, rows, *others)

    monkeypatch.setattr(lgbm_model, "fit_one_weight", spy)
    run_made_up(small_model, tmp_path)
    train_end = small_model.split.train_end
    assert seen and all(len(r) == round(train_end * 0.5) for r in seen)
    assert all(r.max() < train_end for r in seen)  # never a validation or test row


def test_the_threshold_comes_from_validation_and_ignores_test_labels(
    small_model,
) -> None:
    fitted = lgbm_run.fit_best_weight(small_model, np.arange(0, 1800, 2))
    first = lgbm_run.score_both_periods(small_model, fitted)
    changed_truth = small_model.truth.copy()
    changed_truth[small_model.split.test] = 1 - changed_truth[small_model.split.test]
    other = LgbmInputs(
        small_model.features, changed_truth, small_model.split, small_model.names
    )
    second = lgbm_run.score_both_periods(other, fitted)
    assert first["scored"]["threshold"] == second["scored"]["threshold"]
    scores = fitted.model.predict_proba(
        small_model.features[small_model.split.validation]
    )[:, 1]
    expected = best_f1_threshold(
        small_model.truth[small_model.split.validation], scores
    )
    assert first["scored"]["threshold"] == pytest.approx(expected)


def test_the_class_weight_that_wins_is_the_one_with_the_best_validation_pr_auc(
    small_model,
) -> None:
    fitted = lgbm_run.fit_best_weight(small_model, np.arange(0, 1800, 2))
    tried = fitted.validation_pr_auc_by_weight
    assert set(tried) == {"1.0", "10.0", "balanced"}
    best_name = max(tried, key=tried.get)
    expected = lgbm_run.class_weight(
        "balanced" if best_name == "balanced" else float(best_name),
        small_model.truth[np.arange(0, 1800, 2)],
    )
    assert fitted.weight == pytest.approx(expected)


# --- SHAP reason codes ---


def test_top_alerts_get_five_reason_codes_and_a_summary_plot(
    small_model, tmp_path: Path
) -> None:
    pytest.importorskip("shap")
    from src.lgbm_explain import explain_run

    fitted = lgbm_run.fit_best_weight(small_model, np.arange(0, 1800, 2))
    table, plot = explain_run(small_model, fitted.model.booster_, tmp_path, 0.1, 0)
    lines = table.read_text().strip().splitlines()
    assert len(lines) == 1 + config.SHAP_TOP_ALERTS
    header = lines[0].split(",")
    assert header[:4] == ["rank", "row", "score", "is_laundering"]
    assert sum(name.startswith("reason_") for name in header) == 5
    assert plot.stat().st_size > 1000


def test_reason_codes_name_the_strongest_features_first() -> None:
    from src.lgbm_explain import reason_codes

    features = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["a", "b", "c"])
    values = np.array([[0.1, -0.9, 0.5]])
    codes = reason_codes(values, features, 2)
    assert codes == [["b = 2 (-0.90)", "c = 3 (+0.50)"]]


# --- chart ---


def test_the_label_efficiency_chart_gets_a_lightgbm_line(tmp_path: Path) -> None:
    from src.plot import LIGHTGBM_NAME, draw_chart
    from tests.test_collect_plot import make_record, rows_of

    records = [
        make_record(mode, fraction, 0.1 + fraction * 0.1 + gap, seed)
        for seed in (0, 1)
        for fraction in (0.1, 1.0)
        for mode, gap in (("scratch", 0.0), ("finetune", 0.1))
    ]
    baseline = [
        make_record("scratch", fraction, 0.15, seed)
        for seed in (0, 1)
        for fraction in (0.1, 1.0)
    ]
    rows = rows_of(records) + [
        row.__class__(**{**asdict(row), "model": "lightgbm", "run_set": "phaseE"})
        for row in rows_of(baseline)
    ]
    path = tmp_path / "chart.png"
    draw_chart(rows, "test_minority_f1", path, "phaseC")
    import matplotlib.pyplot as plt

    assert path.stat().st_size > 1000 and LIGHTGBM_NAME
    plt.close("all")


# --- extra features (tried on validation only) ---


def extra_frame() -> pd.DataFrame:
    """Account 0 pays 1, 1, then 2; account 1 pays 2 once. One payment per hour."""
    sender = [0, 0, 0, 1]
    receiver = [1, 1, 2, 2]
    stamps = pd.Timestamp("2022-09-01") + pd.to_timedelta([0, 1, 2, 3], unit="h")
    return pd.DataFrame(
        {
            "from_code": sender, "to_code": receiver, "timestamp": stamps,
            "amount_paid": [10.0, 30.0, 50.0, 70.0],
        }
    )  # fmt: skip


def test_extra_features_flag_the_first_payment_of_each_pair() -> None:
    from src.lgbm_features_extra import build_extra_features

    table = build_extra_features(extra_frame())
    assert table["is_first_time_pair"].tolist() == [1.0, 0.0, 1.0, 1.0]


def test_extra_features_count_new_partners_in_a_window() -> None:
    from src.lgbm_features_extra import build_extra_features

    table = build_extra_features(extra_frame())
    # Account 0 met receiver 1 (hour 0) and receiver 2 (hour 2), both inside 24 h.
    assert table["sender_new_partners_24h"].tolist() == [1.0, 1.0, 2.0, 1.0]
    # Receiver 2 was first paid by account 0 (hour 2) and then by account 1 (hour 3).
    assert table["receiver_new_senders_24h"].tolist() == [1.0, 1.0, 1.0, 2.0]


def test_the_second_hop_counts_whom_the_receiver_already_pays() -> None:
    from src.lgbm_features_extra import build_extra_features

    hours = pd.Timestamp("2022-09-01") + pd.to_timedelta([0, 1], unit="h")
    frame = pd.DataFrame(
        {
            "from_code": [1, 0], "to_code": [2, 1], "timestamp": hours,
            "amount_paid": [5.0, 5.0],
        }
    )  # fmt: skip
    table = build_extra_features(frame)
    # At hour 1 account 0 pays account 1, who already paid account 2 (one onward hop).
    assert table["receiver_out_degree"].tolist() == [0.0, 1.0]
    assert table["sender_in_degree"].tolist() == [0.0, 0.0]


def test_amount_history_compares_a_payment_with_the_senders_usual_amount() -> None:
    from src.lgbm_features_extra import build_extra_features

    table = build_extra_features(extra_frame())
    # Sender 0's running mean after payments 10, 30, 50 is 10, 20, 30.
    assert table["sender_mean_amount_sent"].tolist()[:3] == [10.0, 20.0, 30.0]
    assert table["sender_transaction_count"].tolist()[:3] == [1.0, 2.0, 3.0]


def test_extra_features_need_no_label_column() -> None:
    from src.lgbm_features_extra import EXTRA_COLUMNS, build_extra_features

    table = build_extra_features(extra_frame())  # the frame has no label at all
    assert list(table.columns) == EXTRA_COLUMNS
    assert not [name for name in EXTRA_COLUMNS if "laundering" in name]


# --- tuning never reads the test period ---


def test_a_tuning_score_does_not_touch_the_test_rows(small_model) -> None:
    fitted = lgbm_run.fit_best_weight(small_model, np.arange(0, 1800, 2))
    poisoned = small_model.features.copy()
    poisoned[small_model.split.test] = np.nan
    other = LgbmInputs(
        poisoned, small_model.truth, small_model.split, small_model.names
    )
    clean = lgbm_run.score_both_periods(small_model, fitted, include_test=False)
    same = lgbm_run.score_both_periods(other, fitted, include_test=False)
    assert "scored" not in clean and clean == same


def test_the_default_settings_are_project_twos_and_the_tuned_ones_differ() -> None:
    from src.lgbm_model import DEFAULT_SETTINGS, TUNED_SETTINGS

    assert DEFAULT_SETTINGS.leaves == 7 and DEFAULT_SETTINGS.patience == 30
    assert TUNED_SETTINGS.patience > DEFAULT_SETTINGS.patience
    assert 0.5 in TUNED_SETTINGS.weights and TUNED_SETTINGS.leaves == 7


def test_the_record_stores_the_settings_it_was_trained_with(
    small_model, tmp_path: Path
) -> None:
    record = run_made_up(small_model, tmp_path)
    assert record["config"]["lightgbm_settings"]["leaves"] == 7
    assert record["config"]["feature_names"] == list(FEATURE_COLUMNS)
