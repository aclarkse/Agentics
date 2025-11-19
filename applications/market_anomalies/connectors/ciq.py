from __future__ import annotations
import logging
from typing import Dict, Tuple
import pandas as pd

from ._wrds_base import WRDSDataIngestor

logger = logging.getLogger(__name__)

# Core anomaly type IDs (standard CIQ ids)
# 16  = Executive Changes (High Level)
# 81  = M&A Announcement
# 232 = M&A Canceled/Withdrawn (High Impact)
DEFAULT_EVENT_IDS: Tuple[int, ...] = (16, 81, 232)


class CIQIngestor(WRDSDataIngestor):
    """
    Very simple Capital IQ (CIQ) Key Developments ingestor via WRDS.

    Only pulls:
      - Executive changes (high level)
      - M&A announcements
      - M&A canceled/withdrawn

    Returns tidy columns:
      keydevid, companyid, event_date, event_type_id, event_type_name,
      headline, ticker, event_severity, is_exec_change, is_mna_announce, is_mna_canceled
    """

    library = "ciq"

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    # def fetch_keydev_events(
    #     self,
    #     start_date: str,
    #     end_date: str,
    #     event_ids: Tuple[int, ...] = DEFAULT_EVENT_IDS,
    #     only_primary_us_tickers: bool = True,
    # ) -> pd.DataFrame:
    #     """
    #     Parameters
    #     ----------
    #     start_date, end_date : 'YYYY-MM-DD'
    #     event_ids            : CIQ keydeveventtype ids to include (default: 16,81,232)
    #     only_primary_us_tickers : if True, uses wrds_ticker.flag = 1 (primary ticker)
    #     """
    #     if not event_ids:
    #         event_ids = DEFAULT_EVENT_IDS
    #
    #     event_filter = "(" + ",".join(str(i) for i in event_ids) + ")"
    #     primary_flag_clause = "AND t.flag = 1" if only_primary_us_tickers else ""
    #
    #     q = f"""
    #     SELECT
    #         kd.keydevid,
    #         kd.companyid,
    #         kd.announceddate,
    #         kd.keydeveventtypeid,
    #         et.keydeveventtypename,
    #         kd.headline,
    #         t.ticker
    #     FROM {self.library}.wrds_keydev AS kd
    #     JOIN {self.library}.ciqkeydeveventtype AS et
    #          ON kd.keydeveventtypeid = et.keydeveventtypeid
    #     JOIN {self.library}.wrds_ticker AS t
    #          ON kd.companyid = t.companyid
    #     WHERE
    #         kd.keydeveventtypeid IN {event_filter}
    #         AND kd.announceddate BETWEEN '{start_date}' AND '{end_date}'
    #         {primary_flag_clause}
    #     ORDER BY kd.announceddate DESC
    #     """

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_keydev_events(
            self,
            start_date: str,
            end_date: str,
            event_ids: Tuple[int, ...] = DEFAULT_EVENT_IDS,
            only_primary_us_tickers: bool = True,
    ) -> pd.DataFrame:
        """
        Fetches Capital IQ Key Development events within a target time period,
        linking them to primary US tickers and including the Compustat GVKEY.

        Parameters
        ----------
        start_date, end_date : 'YYYY-MM-DD'
        event_ids            : CIQ keydeveventtype ids to include (default: 16, 29, 83)
        only_primary_us_tickers : if True, filters for primary US tickers only.
        """

        if not event_ids:
            # Assuming DEFAULT_EVENT_IDS is defined elsewhere, e.g., (16, 29, 83)
            event_ids = DEFAULT_EVENT_IDS

            # Prepare SQL clauses
        event_filter = "(" + ",".join(str(i) for i in event_ids) + ")"

        # CORRECTED: Use 't.primaryflag' which is confirmed to exist in your wrds_ticker table.
        primary_flag_clause = "AND t.primaryflag = 1" if only_primary_us_tickers else ""

        # We also add an End Date filter on the ticker to ensure the ticker was active
        # during or before the announcement date.
        ticker_date_filter = f"AND (t.enddate IS NULL OR t.enddate >= kd.announcedate)"

        q = f"""
            SELECT 
                kd.keydevid,
                kd.companyid,
                kd.companyname,
                kd.gvkey,
                kd.announcedate AS event_date,
                kd.keydeveventtypeid,
                kd.eventtype,
                kd.headline,
                t.ticker
            FROM {self.library}.wrds_keydev AS kd
            -- Join Ticker link table
            JOIN {self.library}.wrds_ticker AS t
                 ON kd.companyid = t.companyid
            WHERE 
                kd.keydeveventtypeid IN {event_filter}
                AND kd.announcedate BETWEEN '{start_date}' AND '{end_date}'
                AND kd.gvkey IS NOT NULL     -- Only keep records linked to Compustat
                {primary_flag_clause}
                {ticker_date_filter}         -- Filter for tickers active at the event date
            ORDER BY kd.announcedate DESC
            """

        logger.info(
            "CIQ: fetching KeyDev events %s → %s (event_ids=%s)",
            start_date, end_date, event_ids,
        )
        df = self.conn.raw_sql(q, date_cols=['event_date'])
        if df.empty:
            logger.warning("CIQ KeyDev returned 0 rows.")
            return df

        # Normalize column names
        df.rename(
            columns={
                "announcedate": "event_date",
                "keydeveventtypeid": "event_type_id",
                "eventtype": "event_type_name",
            },
            inplace=True,
        )

        # Simple severity heuristic
        severity_map = {16: 0.8, 81: 0.7, 232: 1.0}
        df["event_severity"] = df["event_type_id"].map(severity_map).fillna(0.6)

        # Convenience flags
        df["is_exec_change"]  = (df["event_type_id"] == 16).astype("Int8")
        df["is_mna_announce"] = (df["event_type_id"] == 81).astype("Int8")
        df["is_mna_canceled"] = (df["event_type_id"] == 232).astype("Int8")

        # De-duplicate basic duplicates
        df = df.drop_duplicates(
            subset=["keydevid", "ticker", "event_type_id", "event_date", "headline"]
        )

        logger.info("CIQ: %d KeyDev events after basic filtering.", len(df))
        return df

    @staticmethod
    def schema_doc() -> Dict:
        return {
            "dataset": "CIQ Key Developments (WRDS, basic)",
            "library": "ciq",
            "primary_tables": [
                "wrds_keydev",
                "ciqkeydeveventtype",
                "wrds_ticker",
            ],
            "date_field": "event_date",
            "identifier_fields": ["keydevid", "companyid", "ticker"],
            "fields": {
                "event_type_id": "CIQ KeyDev event type id (e.g., 16, 81, 232)",
                "event_type_name": "Readable event type name",
                "headline": "Event headline from CIQ",
                "event_severity": "Simple severity weight (0.6–1.0)",
                "is_exec_change": "1 if Executive Changes (type_id=16)",
                "is_mna_announce": "1 if M&A Announcement (type_id=81)",
                "is_mna_canceled": "1 if M&A Canceled/Withdrawn (type_id=232)",
            },
            "common_queries": [
                "Executive changes (high level) in last 90 days",
                "M&A announcements by ticker over a window",
                "Canceled / withdrawn deals (high impact) by company",
            ],
        }
