from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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


@dataclass
class IDRBaseline(_Base):
    """Isotonic Distributional Regression (IDR).

    Henzi, Ziegel & Gneiting (2021, JRSS-B), doi:10.1111/rssb.12450: a non-parametric
    estimate of the conditional distribution of the outcome given a covariate, subject
    only to the constraint that it is stochastically monotone in the covariate - no
    parametric family assumed, unlike EMOS's half-normal. The paper's own Section 2.3
    states that "non-parametric isotonic binary regression" emerges as a special case
    of IDR when the outcome is binary - which ``y_bust`` already is, so fitting IDR here
    is exactly isotonic regression of ``y_bust`` on the covariate, via the same
    pool-adjacent-violators machinery ``app/ml/verification.py``'s CORP reliability
    curve (D3) already uses, through ``sklearn.isotonic.IsotonicRegression``.

    Covariate: ``spread_mean`` (falling back to ``spread_max``), the same primary
    covariate EMOS uses - so the comparison is "does a monotone non-parametric fit beat
    a parametric half-normal fit, on the same information?", not a different question
    dressed up as a stronger baseline.
    """

    name: str = "IDR"
    _iso: IsotonicRegression | None = None
    _covariate: str | None = None

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:  # not a logistic model
        return pd.DataFrame(index=ev.index)

    def fit(self, train_events: pd.DataFrame) -> "IDRBaseline":
        y = np.asarray(train_events.get(LABEL, []), int)
        self.base_rate = float(y.mean()) if len(y) else float("nan")
        self._model, self.features = None, []
        self._covariate = next(
            (c for c in ("spread_mean", "spread_max") if c in train_events.columns), None)
        if self._covariate is None or len(np.unique(y)) < 2:
            self._iso = None
            return self

        x = pd.to_numeric(train_events[self._covariate], errors="coerce")
        ok = x.notna().to_numpy()
        if ok.sum() < 50:
            self._iso = None
            return self

        self._iso = IsotonicRegression(out_of_bounds="clip").fit(
            x.to_numpy(float)[ok], y[ok])
        self.features = [self._covariate]
        return self

    def predict_proba(self, events: pd.DataFrame) -> np.ndarray:
        if self._iso is None or self._covariate not in events.columns:
            return _clip(np.full(len(events), self.base_rate))
        x = pd.to_numeric(events[self._covariate], errors="coerce")
        fill = float(np.nanmedian(x)) if x.notna().any() else 0.0
        return _clip(self._iso.predict(x.fillna(fill).to_numpy(float)))


@dataclass
class AnalogBaseline(_Base):
    """Method of analogs: for each case, the empirical bust rate among its k most
    similar training cases, no fitted model at all.

    Hamill & Whitaker (2006, Monthly Weather Review), doi:10.1175/MWR3237.1,
    "Probabilistic Quantitative Precipitation Forecasts Based on Reforecast Analogs" -
    the technique this project's own reforecast archive was built for, and the paper
    that established analog forecasting using a *reforecast* archive specifically
    (rather than a short operational record), which is exactly this project's
    situation. Their k-nearest-neighbour search over forecast-state covariates is
    implemented here directly as ``sklearn.neighbors.KNeighborsClassifier``, whose
    ``predict_proba`` with uniform weighting *is* "the empirical event rate among the k
    nearest neighbours" - no separate vote-counting needed.

    Same covariates as ``LeadSpreadSeasonBaseline`` (lead day, spread, season), so the
    comparison is "does a neighbour lookup beat a parametric logistic fit on the same
    features?" - standardised (z-scored on training data) before the neighbour search,
    since Euclidean distance on raw features would be dominated by whichever one has
    the largest scale (lead_time_days spans 1-10; spread's range depends on the
    variable).

    ``n_neighbors=50`` is a fixed, documented choice in the spirit of Hamill &
    Whitaker's own reforecast-analog practice (dozens of analogs drawn from a large
    historical archive), not tuned against held-out data - the same "fixed by
    definition, not fitted" convention EMOS already uses for its error threshold.
    """

    name: str = "analog"
    n_neighbors: int = 50
    _seasons: list = field(default_factory=list)
    _feature_mean: np.ndarray | None = None
    _feature_std: np.ndarray | None = None
    _knn: KNeighborsClassifier | None = None

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

    def fit(self, train_events: pd.DataFrame) -> "AnalogBaseline":
        y = np.asarray(train_events[LABEL], int)
        self.base_rate = float(y.mean()) if len(y) else float("nan")
        X = self._prepare(train_events, fitting=True)
        self.features = list(X.columns)
        if len(np.unique(y)) < 2 or X.empty or X.shape[1] == 0:
            self._knn = None
            return self

        Xn = X.to_numpy(float)
        self._feature_mean = Xn.mean(axis=0)
        std = Xn.std(axis=0)
        self._feature_std = np.where(std > 0, std, 1.0)
        Xs = (Xn - self._feature_mean) / self._feature_std

        k = max(1, min(self.n_neighbors, len(y)))
        # n_jobs=-1 parallelises the neighbour SEARCH across cores. It changes nothing
        # about the answer - measured on 400,000 fit rows and 50,000 queries in this
        # design's 7 dimensions, the two predict_proba outputs are array_equal - it just
        # stops the search running on one core out of twelve.
        #
        # It matters here because the pooled ladder fits on 13,320,000 rows and queries
        # 2,430,900, which is 6.6x any scale this baseline had run at before. Measured
        # 3.3x on this machine (39.8s -> 12.0s on the 400,000-row case above).
        #
        # Do not extrapolate that cost linearly in the fit size. Measured single-core,
        # holding the query set at 2,430,900 rows and growing only the fit set:
        #
        #     fit rows      query time
        #      200,000        21.0 min
        #      500,000        29.2 min
        #    1,000,000        34.5 min
        #    2,000,000        36.1 min
        #
        # Doubling 1M -> 2M costs 1.6 minutes. The kd-tree makes the per-query cost
        # logarithmic in the fit size, so total runtime is set by the number of QUERIES,
        # not by how much was fitted. An earlier linear extrapolation from these numbers
        # predicted ~18 hours single-threaded and was wrong by an order of magnitude.
        #
        # This is the only acceptable way to make the analog affordable. Subsampling the
        # fit set is not: fewer neighbours to draw on makes a k-NN analog WORSE, so a
        # subsampled analog row understates the baseline and flatters the classifier it is
        # meant to be judged against - silent drift in our own favour, which is what this
        # module's own docstring warns about.
        self._knn = KNeighborsClassifier(
            n_neighbors=k, weights="uniform", n_jobs=-1).fit(Xs, y)
        return self

    def predict_proba(self, events: pd.DataFrame) -> np.ndarray:
        if self._knn is None:
            return _clip(np.full(len(events), self.base_rate))
        X = self._prepare(events, fitting=False)
        Xs = (X.to_numpy(float) - self._feature_mean) / self._feature_std
        proba = self._knn.predict_proba(Xs)
        # predict_proba's columns follow self._knn.classes_, which is only guaranteed
        # to be [0, 1] (and column 1 the bust probability) when fit saw both classes -
        # already required above, but asserted rather than assumed silently.
        assert list(self._knn.classes_) == [0, 1], self._knn.classes_
        return _clip(proba[:, 1])


ALL_BASELINES = (ClimatologyBaseline, LeadDayBaseline, SpreadBaseline,
                 LeadSpreadSeasonBaseline, EMOSBaseline, IDRBaseline, AnalogBaseline)


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
