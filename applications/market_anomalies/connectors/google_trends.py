from ._wrds_base import WRDSDataIngestor
from pytrends.request import TrendReq
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class GoogleTrendsIngestor(WRDSDataIngestor):
    """Public attention signal via Google Trends."""

    def __init__(self, hl="en-US", tz=360):
        self.trends = TrendReq(hl=hl, tz=tz)

    def fetch_interest(self, keywords: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        self.trends.build_payload(keywords, timeframe=f"{start_date} {end_date}")
        df = self.trends.interest_over_time().reset_index()
        return df.drop(columns=["isPartial"], errors="ignore")
