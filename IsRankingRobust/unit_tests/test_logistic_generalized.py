import numpy as np
import pytest

from package.RankAMIP.data_script import simulate_bt_design_matrix
from package.RankAMIP.logistic import LogisticAMIP
from package.RankAMIP.logistic import all_pair_outcome_candidates
from package.RankAMIP.logistic import enumerate_ranking_matchups


def test_all_pair_outcome_candidates_shape_and_labels():
    candidate_X, candidate_y = all_pair_outcome_candidates(4)

    assert candidate_X.shape == (12, 3)
    assert candidate_y.shape == (12,)
    assert set(candidate_y.tolist()) == {0, 1}
    assert np.all(candidate_X[0] == np.array([-1.0, 0.0, 0.0]))
    assert np.all(candidate_X[1] == np.array([-1.0, 0.0, 0.0]))


def test_enumerate_ranking_matchups_all_pairs_returns_all_pairs():
    scores = np.array([0.8, 0.1, -0.2])
    matchups = enumerate_ranking_matchups(scores, candidate_scope="all_pairs")

    assert len(matchups) == 6
    observed_pairs = {(a, b) for a, b, _ in matchups}
    expected_pairs = {
        (0, None),
        (1, None),
        (None, 2),
        (0, 1),
        (0, 2),
        (1, 2),
    }
    assert observed_pairs == expected_pairs


@pytest.mark.parametrize("action", ["drop", "add"])
def test_amip_sign_change_refit_matches_manual_refit(action):
    X, y = simulate_bt_design_matrix(num_teams=8, num_games=400, seed=7)
    amip = LogisticAMIP(X, y, fit_intercept=False, penalty=None)

    kwargs = {}
    if action == "add":
        candidate_X, candidate_y = all_pair_outcome_candidates(8)
        kwargs["candidate_X"] = candidate_X
        kwargs["candidate_y"] = candidate_y

    _, _, original_diff, _, refit_diff, selected = amip.AMIP_sign_change(
        alphaN=3,
        dim_1=0,
        dim_2=1,
        method="1sN",
        refit=True,
        action=action,
        **kwargs,
    )

    full_coef = amip.get_model().coef_[0]
    assert pytest.approx(original_diff, abs=1e-9) == full_coef[0] - full_coef[1]

    if action == "drop":
        keep_mask = np.ones(X.shape[0], dtype=bool)
        keep_mask[selected] = False
        manual_X = X[keep_mask]
        manual_y = y[keep_mask]
    else:
        manual_X = np.vstack([X, candidate_X[selected]])
        manual_y = np.concatenate([y, candidate_y[selected]])

    manual_refit = LogisticAMIP(manual_X, manual_y, fit_intercept=False, penalty=None)
    manual_diff = manual_refit.get_model().coef_[0][0] - manual_refit.get_model().coef_[0][1]
    assert pytest.approx(refit_diff, abs=1e-9) == manual_diff
