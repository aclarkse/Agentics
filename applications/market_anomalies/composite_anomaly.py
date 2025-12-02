from __future__ import annotations
from ticker_resolver import TickerResolver
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import sqlite3
import logging
logger = logging.getLogger(__name__)


# ============================================================
# Config + small Bayesian helper
# ============================================================

@dataclass
class AnomalyConfig:
    """Configuration for CRSP-based anomaly detection."""
    vol_window: int = 6      # works with 12 monthly obs
    ret_lookback: int = 6
    min_obs: int = 3         # don't require 252 obs for monthly data


@dataclass
class ComponentWeights:
    """Weights for composite anomaly score."""
    w_crsp: float = 0.4
    w_compustat: float = 0.25
    w_ibes_eps: float = 0.2
    w_ibes_recs: float = 0.15


class BayesianConfidenceAssessment:
    """
    Tiny helper for mapping z-scores to anomaly probabilities.
    This is intentionally simple; you can swap in your richer
    Bayesian logic later.
    """

    @staticmethod
    def z_to_two_sided_p(z: float) -> float:
        """Approximate two-sided p-value for a z-score."""
        # use a simple normal CDF approximation
        z = float(z)
        # guard
        if np.isnan(z):
            return 1.0
        # tail prob ~ exp(-z^2/2) / (|z|*sqrt(2π)), here we just use exp part
        tail = np.exp(-0.5 * z * z)
        # two-sided, but we squash to [0,1]
        return float(min(1.0, max(0.0, 2 * tail)))

    @staticmethod
    def anomaly_from_p(p: float) -> float:
        """
        Map p-value in [0,1] to anomaly probability in [0,1].
        Small p → high anomaly.
        """
        p = float(np.clip(p, 1e-6, 1.0))
        # simple transform: anomaly = 1 - sqrt(p)
        return float(1.0 - np.sqrt(p))


# ============================================================
# CRSP core: permno-based monthly anomaly features
# ============================================================

class CRSPAnomalyCore:
    """
    Computes CRSP-based anomaly features for a security (by permno):
      - monthly return
      - rolling volatility
      - z-scores vs baseline
      - per-period CRSP anomaly score & normalized price index
    """

    def __init__(self, db_path: str, config: Optional[AnomalyConfig] = None):
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.config = config or AnomalyConfig()
        self.bayes = BayesianConfidenceAssessment()

    def _load_crsp_window(self, permno: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load CRSP monthly data for a permno, including a long history window.

        Assumes SQLite table 'crsp_daily' actually stores your CRSP
        monthly stock file (msf-like) with columns such as:
          - date
          - permno
          - final_ret (preferred) or ret
        """
        if end_date is None:
            with sqlite3.connect(self.db_path) as conn:
                row = pd.read_sql_query("SELECT MAX(date) AS max_date FROM crsp_daily", conn)
            end_date = row["max_date"].iloc[0]

        end_dt = datetime.fromisoformat(str(end_date))
        start_dt = end_dt - timedelta(days=self.config.history_days)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        q = """
        SELECT *
        FROM crsp_daily
        WHERE permno = ?
          AND date BETWEEN ? AND ?
        ORDER BY date
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(q, conn, params=(permno, start_str, end_str))

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df["permno"] = df["permno"].astype(str)

        # Choose return column: prefer final_ret, fallback to ret
        cols_lower = {c.lower(): c for c in df.columns}
        if "final_ret" in cols_lower:
            ret_col = cols_lower["final_ret"]
        elif "ret" in cols_lower:
            ret_col = cols_lower["ret"]
        else:
            raise ValueError("CRSP table has no 'final_ret' or 'ret' column.")

        df["ret"] = pd.to_numeric(df[ret_col], errors="coerce")

        return df

    def compute_crsp_features(self, permno: str | int) -> pd.DataFrame:
        """
        Pull CRSP time series for a given permno from `crsp_daily`
        and compute simple anomaly features (rolling volatility + z-scores).

        We build a synthetic price index from returns.
        """
        p_int = int(permno)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT date, permno, ret
                FROM crsp_daily
                WHERE permno = ?
                ORDER BY date
                """,
                conn,
                params=(p_int,),
            )

        if df.empty:
            return df

        # Ensure datetime + sorted
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Clean returns
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0)

        # --- Synthetic price index (starts at 1.0) ---
        df["price_index"] = (1.0 + df["ret"]).cumprod()

        # If too few observations, return basic series with neutral anomaly
        if len(df) < self.config.min_obs:
            df["volatility"] = np.nan
            df["ret_z"] = np.nan
            df["vol_z"] = np.nan
            df["anomaly_crsp"] = 0.5
            return df

        # Rolling volatility
        df["volatility"] = (
            df["ret"]
            .rolling(self.config.vol_window, min_periods=self.config.min_obs)
            .std()
        )

        # z-scores
        vol_mean = df["volatility"].mean(skipna=True)
        vol_std = df["volatility"].std(skipna=True)
        df["vol_z"] = (df["volatility"] - vol_mean) / (vol_std + 1e-10)

        ret_mean = df["ret"].mean(skipna=True)
        ret_std = df["ret"].std(skipna=True)
        df["ret_z"] = (df["ret"] - ret_mean) / (ret_std + 1e-10)

        # map vol_z into [0,1] anomaly-ish score
        df["anomaly_crsp"] = 0.5 + 0.5 * np.tanh(df["vol_z"].fillna(0.0))

        return df


# ============================================================
# Composite detector using master_db.parquet
# ============================================================

class CompositeAnomalyDetector:
    ...

    def __init__(
        self,
        db_path: str,
        master_parquet_path: Optional[str] = None,
        config: Optional[AnomalyConfig] = None,
        weights: Optional[ComponentWeights] = None,
    ):
        # Convert to absolute path ALWAYS
        self.db_path = str(Path(db_path).expanduser().resolve())

        # Parent dir (project root discovery)
        db_dir = Path(self.db_path).resolve().parent

        self.config = config or AnomalyConfig()
        self.weights = weights or ComponentWeights()
        self.crsp_core = CRSPAnomalyCore(self.db_path, self.config)

        # --- Centralized CRSP ID resolver ---
        self.ticker_resolver = TickerResolver(
            self.db_path,
            master_parquet_path
        )

        # to have access to the raw master mapping
        self.master = self.ticker_resolver.master


    # ---------------------------
    # ID resolution
    # ---------------------------
    def resolve_ids(self, ticker: str) -> Dict[str, Optional[str]]:
        """
        Resolve permno / gvkey / cusip from a ticker.

        Strategy:
          - CRSP master_db (via TickerResolver) → permno, cusip
          - Compustat (if it has 'ticker') → gvkey
          - IBES (if it has 'ticker' & 'cusip') → cusip fallback
        """
        t = ticker.strip().upper()

        # --- 1) CRSP: ticker → permno, cusip via master_db ---
        ids_crsp = self.ticker_resolver.resolve(t)
        ids: Dict[str, Optional[str]] = {
            "ticker": t,
            "permno": ids_crsp.get("permno"),
            "cusip": ids_crsp.get("cusip"),
            "gvkey": None,
        }

        # --- 2) Compustat: ticker → gvkey (only if 'ticker' column exists) ---
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(compustat_quarterly);")
                comp_cols = [row[1].lower() for row in cur.fetchall()]

                if "gvkey" in comp_cols and "ticker" in comp_cols:
                    comp_map = pd.read_sql_query(
                        """
                        SELECT DISTINCT gvkey
                        FROM compustat_quarterly
                        WHERE UPPER(ticker) = ?
                        ORDER BY datadate DESC
                        LIMIT 1
                        """,
                        conn,
                        params=(t,),
                    )
                    if not comp_map.empty:
                        ids["gvkey"] = str(comp_map.loc[0, "gvkey"])
                else:
                    logger.debug(
                        "[resolve_ids] compustat_quarterly has no 'ticker' column; "
                        "skipping gvkey lookup from Compustat."
                    )
        except Exception as e:
            print(f"[resolve_ids] compustat warning: {e}")

        # --- 3) IBES EPS: ticker → cusip (only if both columns exist) ---
        if ids["cusip"] is None:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.cursor()
                    cur.execute("PRAGMA table_info(ibes_eps_summary);")
                    ibes_cols = [row[1].lower() for row in cur.fetchall()]

                    if "ticker" in ibes_cols and "cusip" in ibes_cols:
                        ibes_map = pd.read_sql_query(
                            """
                            SELECT DISTINCT cusip
                            FROM ibes_eps_summary
                            WHERE UPPER(ticker) = ?
                            LIMIT 1
                            """,
                            conn,
                            params=(t,),
                        )
                        if not ibes_map.empty:
                            ids["cusip"] = str(ibes_map.loc[0, "cusip"])
                    else:
                        print("[resolve_ids] ibes_eps_summary has no 'ticker'/'cusip' "
                              "columns; skipping cusip lookup from IBES.")
            except Exception as e:
                print(f"[resolve_ids] ibes lookup warning: {e}")

        return ids


    # ---------------------------
    # Dataset-specific loaders
    # ---------------------------

    def _load_latest_compustat(self, gvkey: Optional[str]) -> Optional[pd.Series]:
        if gvkey is None:
            return None
        q = """
        SELECT *
        FROM compustat_quarterly
        WHERE gvkey = ?
        ORDER BY datadate DESC
        LIMIT 1
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(q, conn, params=(gvkey,))
        if df.empty:
            return None
        return df.iloc[0]

    def _load_latest_ibes_eps(self, ticker: str) -> Optional[pd.Series]:
        """
        Load the most recent IBES EPS summary row for a given ticker,
        based on `estimate_date` in `ibes_eps_summary`.
        """
        t = ticker.strip().upper()

        q = """
            SELECT *
            FROM ibes_eps_summary
            WHERE UPPER(ticker) = ?
            ORDER BY estimate_date DESC LIMIT 1 \
            """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(q, conn, params=(t,))

        if df.empty:
            return None

        return df.iloc[0]

    def _load_recent_ibes_recs(
            self, ticker: str, window_days: int = 365
    ) -> Optional[pd.DataFrame]:
        """
        Load recent IBES recommendations for a ticker from ibes_recommendations.

        We DO NOT assume the date column is called 'anndats'.
        We inspect the table and pick a likely date column:
        - 'anndats'          (raw WRDS)
        - 'announce_date'    (common rename)
        - 'recommendation_date'
        - 'date'
        """
        t = ticker.strip().upper()

        with sqlite3.connect(self.db_path) as conn:
            info = pd.read_sql_query("PRAGMA table_info(ibes_recommendations);", conn)
            colnames = {c.lower(): c for c in info["name"]}

            # Pick a date column
            if "anndats" in colnames:
                date_col = colnames["anndats"]
            elif "announce_date" in colnames:
                date_col = colnames["announce_date"]
            elif "recommendation_date" in colnames:
                date_col = colnames["recommendation_date"]
            elif "date" in colnames:
                date_col = colnames["date"]
            else:
                # No obvious date column → just return all recs for this ticker
                df = pd.read_sql_query(
                    "SELECT * FROM ibes_recommendations WHERE UPPER(ticker) = ?",
                    conn,
                    params=(t,),
                )
                return df if not df.empty else None

            # Find the latest date for this ticker
            q_max = f"""
                SELECT MAX({date_col}) AS max_date
                FROM ibes_recommendations
                WHERE UPPER(ticker) = ?
            """
            max_df = pd.read_sql_query(q_max, conn, params=(t,))
            if max_df.empty or max_df["max_date"].isna().all():
                return None

            latest = pd.to_datetime(max_df.loc[0, "max_date"])
            cutoff = latest - pd.Timedelta(days=window_days)

            # Pull window of recs
            q = f"""
                SELECT *
                FROM ibes_recommendations
                WHERE UPPER(ticker) = ?
                  AND {date_col} BETWEEN ? AND ?
                ORDER BY {date_col} DESC
            """
            df = pd.read_sql_query(
                q,
                conn,
                params=(
                    t,
                    cutoff.strftime("%Y-%m-%d"),
                    latest.strftime("%Y-%m-%d"),
                ),
            )

        return df if not df.empty else None

    def _load_recent_ibes_recs(self, ticker: str, window_days: int = 180) -> Optional[pd.DataFrame]:
        """
        Load IBES recommendation records for a ticker within a recent time window.
        We DO NOT assume the column is called 'anndats' — we detect the correct date column.
        """
        t = ticker.strip().upper()

        with sqlite3.connect(self.db_path) as conn:
            # --- 1) Inspect schema ---
            info = pd.read_sql_query("PRAGMA table_info(ibes_recommendations);", conn)
            colnames = {c.lower(): c for c in info["name"]}

            # --- 2) Choose best available date column ---
            if "anndats" in colnames:
                date_col = colnames["anndats"]
            elif "announce_date" in colnames:
                date_col = colnames["announce_date"]
            elif "recommendation_date" in colnames:
                date_col = colnames["recommendation_date"]
            elif "date" in colnames:
                date_col = colnames["date"]
            else:
                # No recognizable date column → fallback: return all recs for ticker
                df = pd.read_sql_query(
                    "SELECT * FROM ibes_recommendations WHERE UPPER(ticker) = ?",
                    conn,
                    params=(t,),
                )
                return df if not df.empty else None

            # --- 3) Determine available max date ---
            q_max = f"""
                SELECT MAX({date_col}) AS max_date
                FROM ibes_recommendations
                WHERE UPPER(ticker) = ?
            """
            max_row = pd.read_sql_query(q_max, conn, params=(t,))
            max_date = max_row["max_date"].iloc[0]

            if max_date is None:
                return None

            end_date = pd.to_datetime(max_date)
            start_date = end_date - timedelta(days=window_days)

            # --- 4) Pull window of recs ---
            q = f"""
                SELECT *
                FROM ibes_recommendations
                WHERE UPPER(ticker) = ?
                  AND {date_col} BETWEEN ? AND ?
                ORDER BY {date_col} DESC
            """

            df = pd.read_sql_query(
                q,
                conn,
                params=(
                    t,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                ),
            )

        return df if not df.empty else None

    # ---------------------------
    # Component scoring helpers
    # ---------------------------

    @staticmethod
    def _logistic(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _score_crsp_from_row(row: pd.Series) -> float:
        """
        Turn CRSP anomalies into a single [0,1] score.

        Uses vol_z as main driver; magnitude of z == anomaly.
        """
        z = float(row.get("vol_z", 0.0))
        if np.isnan(z):
            return 0.5
        # use magnitude of z (two-sided)
        mag = abs(z)
        # scale: mag=0 → ~0, mag=2 → ~0.6, mag=3 → ~0.8
        score = 1.0 - np.exp(-0.5 * (mag ** 2) / (2.0 ** 2))
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _score_compustat(row: Optional[pd.Series]) -> float:
        """
        Score fundamentals anomaly based on margin z-scores if available.

        We treat *negative* margin z-scores as more anomalous (weakness).
        """
        if row is None:
            return 0.5

        # Try a few likely column names
        for col in ["net_margin_zscore", "net_margin_z", "is_net_margin_anomaly"]:
            if col in row.index:
                val = row[col]
                if pd.isna(val):
                    continue
                # if it's a 0/1 anomaly flag, just map directly
                if col.startswith("is_"):
                    return float(np.clip(val, 0.0, 1.0))
                z = float(val)
                # negative z → high anomaly, use logistic
                return float(1.0 / (1.0 + np.exp(1.0 * z)))  # z << 0 → score ~1

        # fallback if we don't find any margin z-score
        return 0.5

    @staticmethod
    def _score_ibes_eps(row: Optional[pd.Series]) -> Optional[float]:
        """
        Turn the latest EPS row into an anomaly score in [0,1].

        Heuristic:
        - earnings_surprise = (actual - consensus) / |consensus|
        - Large negative surprise => high anomaly
        - Large positive surprise => lower anomaly
        - Higher analyst coverage => more confident score
        """
        if row is None:
            return None

        cons = row.get("consensus_estimate")
        actual = row.get("actual_eps")

        if pd.isna(cons) or pd.isna(actual):
            return None

        surprise = (actual - cons) / (abs(cons) + 1e-10)
        coverage = row.get("num_analysts", 0.0) or 0.0

        # Raw anomaly: negative surprise -> high score, positive -> low
        raw = -surprise  # flip: big negative surprise => big positive raw
        # squash into [0,1] via tanh
        base_score = 0.5 + 0.5 * np.tanh(raw * 2.0)  # scale factor 2.0 is tunable

        # Coverage factor: with more analysts, lean more on this signal
        cov_weight = float(np.clip(coverage / 10.0, 0.0, 1.0))
        # Blend toward neutral (0.5) if coverage is low
        score = (1 - cov_weight) * 0.5 + cov_weight * base_score

        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _score_ibes_recs(df: Optional[pd.DataFrame]) -> Optional[float]:
        """
        Turn a recent recommendations window into an anomaly score in [0,1].

        Heuristic:
          - recommendation_code: 1=Strong Buy ... 5=Sell (higher = worse)
          - More bearish on average => higher anomaly.
        """
        if df is None or df.empty:
            return None

        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}

        # 1) Try exact / strong matches first
        if "recommendation_code" in lower_map:
            rec_col = lower_map["recommendation_code"]
        elif "ireccd" in lower_map:
            rec_col = lower_map["ireccd"]
        elif "reccd" in lower_map:
            rec_col = lower_map["reccd"]
        else:
            # 2) Fallback: any column with "code" in the name, but NOT date / ticker
            candidates = [
                c
                for c in cols
                if ("code" in c.lower())
                   and ("date" not in c.lower())
                   and ("time" not in c.lower())
                   and ("ticker" not in c.lower())
            ]
            if not candidates:
                return None
            rec_col = candidates[0]

        # Convert that column to numeric safely
        codes = pd.to_numeric(df[rec_col], errors="coerce").dropna()
        if codes.empty:
            return None

        # Map average code from [1,5] -> [0,1]
        avg_code = codes.mean()
        score = (avg_code - 1.0) / 4.0
        return float(np.clip(score, 0.0, 1.0))

    # ---------------------------
    # Public API
    # ---------------------------

    def compute_composite_for_ticker(
            self, ticker: str
    ) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
        """
        Given a TICKER, build:
          - CRSP time series (if available),
          - Compustat / IBES static components (if available),
          - a composite anomaly score in [0,1],
          - a components dict for interpretability.

        Never hard-fails just because one component is missing.
        """
        ids = self.resolve_ids(ticker)
        permno = ids.get("permno")
        gvkey = ids.get("gvkey")
        resolved_ticker = ids.get("ticker")

        # --------------------------------------------------
        # 1) CRSP time series (OPTIONAL)
        # --------------------------------------------------
        ts = pd.DataFrame()
        crsp_score = 0.5  # neutral if no CRSP
        crsp_details: Dict[str, Any] = {
            "latest_date": None,
            "vol_z": None,
            "ret_z": None,
            "anomaly_crsp": None,
            "note": "CRSP not used (no permno mapping or no time series rows).",
        }

        if permno is not None:
            ts_candidate = self.crsp_core.compute_crsp_features(permno)
            if ts_candidate is not None and not ts_candidate.empty:
                ts = ts_candidate
                latest_row = ts.iloc[-1]
                # helper you define; simple mapping from vol_z to [0,1]
                crsp_score = float(
                    0.5 + 0.5 * np.tanh(latest_row.get("vol_z", 0.0))
                )
                crsp_details = {
                    "latest_date": str(
                        getattr(latest_row["date"], "date", lambda: latest_row["date"])()
                    ),
                    "vol_z": float(latest_row.get("vol_z", np.nan)),
                    "ret_z": float(latest_row.get("ret_z", np.nan)),
                    "anomaly_crsp": crsp_score,
                    "note": None,
                }
            else:
                crsp_details["note"] = f"No CRSP rows for permno {permno} in crsp_daily."

        # --------------------------------------------------
        # 2) Compustat (OPTIONAL, via gvkey)
        # --------------------------------------------------
        comp_row = self._load_latest_compustat(gvkey) if gvkey is not None else None
        comp_score = self._score_compustat(comp_row)

        # --------------------------------------------------
        # 3) IBES EPS (OPTIONAL, via ticker)
        # --------------------------------------------------
        ibes_eps_row = (
            self._load_latest_ibes_eps(resolved_ticker) if resolved_ticker else None
        )
        ibes_eps_score = self._score_ibes_eps(ibes_eps_row)

        # --------------------------------------------------
        # 4) IBES Recs (OPTIONAL, via ticker)
        # --------------------------------------------------
        ibes_recs_df = (
            self._load_recent_ibes_recs(resolved_ticker) if resolved_ticker else None
        )
        ibes_recs_score = self._score_ibes_recs(ibes_recs_df)

        # --------------------------------------------------
        # 5) Composite score = weighted average of AVAILABLE components
        # --------------------------------------------------
        w = self.weights
        scores = {
            "crsp": crsp_score,
            "compustat": comp_score,
            "ibes_eps": ibes_eps_score,
            "ibes_recs": ibes_recs_score,
        }
        weights = {
            "crsp": w.w_crsp,
            "compustat": w.w_compustat,
            "ibes_eps": w.w_ibes_eps,
            "ibes_recs": w.w_ibes_recs,
        }

        used_scores = []
        used_weights = []
        for k, s in scores.items():
            if s is None or (isinstance(s, float) and np.isnan(s)):
                continue
            used_scores.append(s)
            used_weights.append(weights[k])

        if used_scores:
            latest_composite = float(np.average(used_scores, weights=used_weights))
        else:
            latest_composite = 0.5  # shrug

        # --------------------------------------------------
        # 6) Per-date composite if we do have CRSP series
        # --------------------------------------------------
        if not ts.empty:
            ts = ts.copy()
            crsp_series = ts.get("anomaly_crsp", pd.Series(0.5, index=ts.index)).fillna(
                0.5
            )
            total_w = w.w_crsp + w.w_compustat + w.w_ibes_eps + w.w_ibes_recs
            ts["composite_score"] = (
                                            w.w_crsp * crsp_series
                                            + w.w_compustat * (comp_score if comp_score is not None else 0.5)
                                            + w.w_ibes_eps * (ibes_eps_score if ibes_eps_score is not None else 0.5)
                                            + w.w_ibes_recs * (
                                                ibes_recs_score if ibes_recs_score is not None else 0.5
                                            )
                                    ) / total_w

        components: Dict[str, Any] = {
            "ids": ids,
            "crsp": crsp_details,
            "compustat": {
                "score": float(comp_score) if comp_score is not None else None,
                "raw_row": comp_row.to_dict()
                if isinstance(comp_row, pd.Series)
                else None,
            },
            "ibes_eps": {
                "score": float(ibes_eps_score)
                if ibes_eps_score is not None
                else None,
                "raw_row": ibes_eps_row.to_dict()
                if isinstance(ibes_eps_row, pd.Series)
                else None,
            },
            "ibes_recs": {
                "score": float(ibes_recs_score)
                if ibes_recs_score is not None
                else None,
                "num_records": int(len(ibes_recs_df))
                if isinstance(ibes_recs_df, pd.DataFrame)
                else 0,
            },
            "weights": {
                "crsp": w.w_crsp,
                "compustat": w.w_compustat,
                "ibes_eps": w.w_ibes_eps,
                "ibes_recs": w.w_ibes_recs,
            },
            "latest_composite_score": float(latest_composite),
        }

        return ts, latest_composite, components

