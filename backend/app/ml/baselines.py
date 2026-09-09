from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

LABEL = "y_bust"
_EPS = 1e-6

_LOGIT = dict(max_iter=1000, solver="lbfgs", random_state=42)


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)


@dataclass
class _Base:

    name: str = "baseline"
    features: list = field(default_factory=list)
    base_rate: float = float("nan")
    _model: LogisticRegression | None = None
    _medians: dict = field(default_factory=dict)
    _columns: list = field(default_factory=list)

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _prepare(self, ev: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        X = self._design(ev)
        if fitting:
            self._medians = {c: float(X[c].median()) for c in X.columns
                             if X[c].notna().any()}
        X = X.copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(self._medians.get(c, 0.0))
        if fitting:
            self._columns = list(X.columns)
        return X.reindex(columns=self._columns, fill_value=0.0)

    def fit(self, train_events: pd.DataFrame) -> "_Base":
        y = np.asarray(train_events[LABEL], int)
        self.base_rate = float(y.mean()) if len(y) else float("nan")
        X = self._prepare(train_events, fitting=True)
        self.features = list(X.columns)
        if len(np.unique(y)) < 2 or X.empty or X.shape[1] == 0:
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

    name: str = "lead_day"

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"lead_time_days": ev["lead_time_days"]}, index=ev.index)


@dataclass
class SpreadBaseline(_Base):

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


@dataclass
class EMOSBaseline(_Base):
    """Ensemble Model Output Statistics / Non-homogeneous Gaussian Regression.

    Gneiting et al. (2005), "Calibrated probabilistic forecasting using ensemble model
    output statistics and minimum CRPS estimation" - the standard statistical
    post-processing for ensemble forecasts, and the comparison anyone who works with
    ensembles will ask for. Without it, the ladder is this project beating four baselines
    of its own design.

    What makes it EMOS rather than another logistic regression on spread: the ensemble
    spread sets the **scale of a predictive distribution**, it is not a linear term in a
    probability. That is the "non-homogeneous" in NGR, and it is the difference from
    SpreadBaseline, which uses the same column as an ordinary feature.

    Per variable, the error magnitude is modelled as half-normal with scale

        sigma_v = a_v + b_v * spread_v

    fitted by maximum likelihood on training rows. For a half-normal,
    E|e| = sigma * sqrt(2/pi), so the fit is a least-squares regression of observed |error|
    on spread, rescaled - closed form, no optimiser to converge or fail silently. The bust
    probability follows analytically:

        P(bust) = 1 - prod_v erf( threshold_v / (sigma_v * sqrt(2)) )

    **Stated rather than hidden:** that product assumes the variables are independent, and
    they are not - a moist, unstable day misses on rainfall and humidity together. The
    assumption makes this baseline slightly optimistic about how often nothing busts. It
    is a baseline, and the alternative is fitting a full error covariance on the same data
    the model above it uses, which would make it something other than a baseline.

    Thresholds are the 90th percentile of each variable's own error on **training rows
    only**, which is the project's own definition of a bust.
    """

    name: str = "EMOS"
    thresholds_: dict = field(default_factory=dict)
    scales_: dict = field(default_factory=dict)
    dependence_: float = 1.0

    @staticmethod
    def _variables(ev: pd.DataFrame) -> list:
        """Variables with both an observed error and an ensemble spread."""
        errs = {c[len("actual_err_"):] for c in ev.columns if c.startswith("actual_err_")}
        spr = {c[len("spread_"):] for c in ev.columns if c.startswith("spread_")}
        return sorted(errs & spr - {"mean", "max"})

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:  # not a logistic model
        return pd.DataFrame(index=ev.index)

    def fit(self, train_events: pd.DataFrame) -> "EMOSBaseline":
        y = np.asarray(train_events.get(LABEL, []), int)
        self.base_rate = float(y.mean()) if len(y) else float("nan")
        self._model, self.features = None, []
        self.thresholds_, self.scales_ = {}, {}

        for var in self._variables(train_events):
            err = pd.to_numeric(train_events[f"actual_err_{var}"], errors="coerce")
            spr = pd.to_numeric(train_events[f"spread_{var}"], errors="coerce")
            ok = err.notna() & spr.notna()
            if ok.sum() < 50:
                continue
            e, s = err[ok].to_numpy(float), spr[ok].to_numpy(float)
            self.thresholds_[var] = float(np.percentile(e, 90.0))

            # E|e| = sigma*sqrt(2/pi) for a half-normal, so regressing |e| on spread and
            # rescaling by sqrt(pi/2) recovers the scale directly.
            k = np.sqrt(np.pi / 2.0)
            design = np.column_stack([np.ones_like(s), s])
            coef, *_ = np.linalg.lstsq(design, e * k, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            # A scale must be positive: a flat or falling fit degenerates to a constant
            # rather than producing a negative sigma at large spread.
            floor = float(np.mean(e) * k) * 1e-3 or _EPS
            self.scales_[var] = (max(a, floor), max(b, 0.0))
            self.features.append(f"spread_{var}")

        self.dependence_ = self._fit_dependence(train_events, y)
        return self

    def _fit_dependence(self, train_events: pd.DataFrame, y: np.ndarray) -> float:
        """One parameter absorbing the correlation between variables.

        Treating the variables as independent over-forecasts: eight variables each
        exceeding their own 90th percentile 10% of the time gives 1 - 0.9^8 = 0.570,
        against an observed bust rate of 0.434 on the real data. The ranking survives that
        - it is a monotone transform - but the Brier score does not.

        So P(bust) = 1 - (prod P_no)^gamma, with gamma fitted on training rows by
        minimising Brier. gamma below 1 means the variables move together, and
        `gamma * len(scales_)` is the effective number of independent variables, which is
        a quantity worth reporting rather than a fudge factor.
        """
        if not self.scales_ or len(y) == 0:
            return 1.0
        p_no = self._product_no_bust(train_events)
        if p_no is None:
            return 1.0
        grid = np.linspace(0.05, 1.0, 40)
        briers = [np.mean((_clip(1.0 - p_no ** g) - y) ** 2) for g in grid]
        return float(grid[int(np.argmin(briers))])

    def _product_no_bust(self, events: pd.DataFrame):
        """prod_v P(|e_v| <= threshold_v), or None if no variable is usable here."""
        from math import erf

        usable = [v for v in self.scales_
                  if f"spread_{v}" in events.columns and v in self.thresholds_]
        if not usable:
            return None
        vec_erf = np.vectorize(erf)
        out = np.ones(len(events), dtype=float)
        for var in usable:
            a, b = self.scales_[var]
            spr = pd.to_numeric(events[f"spread_{var}"], errors="coerce").to_numpy(float)
            med = np.nanmedian(spr)
            spr = np.nan_to_num(spr, nan=float(med) if np.isfinite(med) else 0.0)
            sigma = np.maximum(a + b * spr, _EPS)
            out *= vec_erf(self.thresholds_[var] / (sigma * np.sqrt(2.0)))
        return out

    def predict_proba(self, events: pd.DataFrame) -> np.ndarray:
        p_no = self._product_no_bust(events)
        if p_no is None:
            # Nothing to condition on. The honest answer is climatology, not a guess.
            return _clip(np.full(len(events), self.base_rate))
        return _clip(1.0 - p_no ** self.dependence_)


ALL_BASELINES = (ClimatologyBaseline, LeadDayBaseline, SpreadBaseline,
                 LeadSpreadSeasonBaseline, EMOSBaseline)


def brier(y_true, proba) -> float:
    y = np.asarray(y_true, float)
    p = np.asarray(proba, float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def brier_skill_score(y_true, proba, reference_proba) -> float:
    bs_ref = brier(y_true, reference_proba)
    if not np.isfinite(bs_ref) or bs_ref <= 0:
        return float("nan")
    return float(1.0 - brier(y_true, proba) / bs_ref)


def fit_all(train_events: pd.DataFrame) -> dict:
    out = {}
    for cls in ALL_BASELINES:
        m = cls().fit(train_events)
        out[m.name] = m
    return out
