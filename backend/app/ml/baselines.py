"""Baseline bust-probability models — the "compared to what?" for the classifier.

A ROC-AUC on its own answers nothing. Forecast error grows with lead time, so a model
whose only input is the lead day already scores well above chance; that is the comparison
a reader has in mind, and beating it is the actual claim. These four make the claim
measurable, cheapest first:

    ClimatologyBaseline        the training base rate, for every row. The reference
                               forecast for the Brier skill score.
    LeadDayBaseline            lead day only. The one a meteorologist will ask about.
    SpreadBaseline             ensemble spread only. The field's standard cheap proxy
                               for forecast uncertainty.
    LeadSpreadSeasonBaseline   all three, to show what a sensible linear model gets
                               before any of the per-variable regressor machinery.

Every one is *fit on the training split only* — base rate, imputation medians, scaling
and coefficients alike — and then evaluated on rows it has never seen. They are scored by
`scripts/run_baselines.py` on the same event frames the classifier is scored on, so no
split, label or threshold is ever recomputed here.

Nothing in this module is imported by the serving path; it exists to be reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

LABEL = "y_bust"
# Bounded away from 0 and 1: a degenerate training split (no busts, or nothing but) would
# otherwise produce a Brier score of exactly 0 for the reference forecast and an infinite
# skill score for everything measured against it.
_EPS = 1e-6

_LOGIT = dict(max_iter=1000, solver="lbfgs", random_state=42)


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)


@dataclass
class _Base:
    """Common shape: fit(train_events) -> self, predict_proba(events) -> np.ndarray."""

    name: str = "baseline"
    features: list = field(default_factory=list)
    base_rate: float = float("nan")
    _model: LogisticRegression | None = None
    _medians: dict = field(default_factory=dict)
    _columns: list = field(default_factory=list)

    # -- feature construction, identical at fit and predict time -----------------
    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _prepare(self, ev: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        X = self._design(ev)
        if fitting:
            # Medians come from train and are then reused verbatim: imputing a test row
            # with a statistic of the test set would leak.
            self._medians = {c: float(X[c].median()) for c in X.columns
                             if X[c].notna().any()}
        X = X.copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(self._medians.get(c, 0.0))
        if fitting:
            self._columns = list(X.columns)
        # Reindex so a category absent from one split cannot shift the column order.
        return X.reindex(columns=self._columns, fill_value=0.0)

    def fit(self, train_events: pd.DataFrame) -> "_Base":
        y = np.asarray(train_events[LABEL], int)
        self.base_rate = float(y.mean()) if len(y) else float("nan")
        X = self._prepare(train_events, fitting=True)
        self.features = list(X.columns)
        if len(np.unique(y)) < 2 or X.empty or X.shape[1] == 0:
            # Nothing to separate; fall back to the base rate. Reported, not silent.
            self._model = None
            return self
        self._model = LogisticRegression(**_LOGIT).fit(X.to_numpy(float), y)
        return self

    def predict_proba(self, events: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return _clip(np.full(len(events), self.base_rate))
        X = self._prepare(events, fitting=False)
        return _clip(self._model.predict_proba(X.to_numpy(float))[:, 1])


@dataclass
class ClimatologyBaseline(_Base):
    """Predict the training-set bust rate for every row. The BSS reference."""

    name: str = "climatology"

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=ev.index)

    def fit(self, train_events: pd.DataFrame) -> "ClimatologyBaseline":
        y = np.asarray(train_events[LABEL], int)
        self.base_rate = float(y.mean()) if len(y) else float("nan")
        self._model, self.features = None, []
        return self

    def predict_proba(self, events: pd.DataFrame) -> np.ndarray:
        return _clip(np.full(len(events), self.base_rate))


@dataclass
class LeadDayBaseline(_Base):
    """Logistic regression on lead day alone."""

    name: str = "lead_day"

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"lead_time_days": ev["lead_time_days"]}, index=ev.index)


@dataclass
class SpreadBaseline(_Base):
    """Logistic regression on ensemble spread alone."""

    name: str = "spread"

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for c in ("spread_mean", "spread_max"):
            if c in ev.columns:
                cols[c] = ev[c]
        if not cols:
            return pd.DataFrame(index=ev.index)
        return pd.DataFrame(cols, index=ev.index)


@dataclass
class LeadSpreadSeasonBaseline(_Base):
    """Lead day + spread + season, season one-hot encoded against the train categories."""

    name: str = "lead+spread+season"
    _seasons: list = field(default_factory=list)

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        cols = {"lead_time_days": ev["lead_time_days"]}
        for c in ("spread_mean", "spread_max"):
            if c in ev.columns:
                cols[c] = ev[c]
        X = pd.DataFrame(cols, index=ev.index)
        if "season" in ev.columns:
            s = ev["season"].astype(str)
            if not self._seasons:
                self._seasons = sorted(s.dropna().unique())
            for season in self._seasons:
                X[f"season_{season}"] = (s == season).astype(float)
        return X


ALL_BASELINES = (ClimatologyBaseline, LeadDayBaseline, SpreadBaseline,
                 LeadSpreadSeasonBaseline)


def brier(y_true, proba) -> float:
    y = np.asarray(y_true, float)
    p = np.asarray(proba, float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def brier_skill_score(y_true, proba, reference_proba) -> float:
    """1 - BS/BS_ref. Positive means better than the reference; 0 means no skill over it.

    The reference is climatology fitted on train, so this is the standard "does it beat
    knowing nothing but the base rate" question, asked on held-out rows.
    """
    bs_ref = brier(y_true, reference_proba)
    if not np.isfinite(bs_ref) or bs_ref <= 0:
        return float("nan")
    return float(1.0 - brier(y_true, proba) / bs_ref)


def fit_all(train_events: pd.DataFrame) -> dict:
    """Fit every baseline on the training split. Returns {name: fitted model}."""
    out = {}
    for cls in ALL_BASELINES:
        m = cls().fit(train_events)
        out[m.name] = m
    return out
