import logging
import os
from datetime import datetime
from pathlib import Path

from connectors.crsp import CRSPIngestor
from connectors.compustat import CompustatIngestor
from connectors.ibes import IBESIngestor
from connectors.sdc import SDCIngestor
from connectors.boardex import BoardExIngestor
from connectors.fred import FREDIngestor
from connectors.yahoo_finance import YahooFinanceIngestor
from connectors.google_trends import GoogleTrendsIngestor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WRDS_USER = os.getenv("WRDS_USERNAME", "your_wrds_username")
FRED_KEY  = os.getenv("FRED_API_KEY")

DATA_DIR = Path(__file__).resolve().parent / "data"

def main():
    logger.info("=== Market Anomalies Data Ingestion Started ===")
    start_time = datetime.now()

    # CRSP (market data)
    with CRSPIngestor(WRDS_USER) as crsp:
        s, e = crsp.get_date_range(90)
        df = crsp.fetch_daily_stock_data(s, e)
        crsp.save_data(df, "wrds/crsp_daily")
        crsp.save_schema(crsp.get_schema_documentation(), "wrds_crsp")

    # Compustat (fundamentals)
    with CompustatIngestor(WRDS_USER) as comp:
        s, e = comp.get_date_range(180)
        df = comp.fetch_quarterly_fundamentals(s, e)
        comp.save_data(df, "wrds/compustat_quarterly")
        comp.save_schema(comp.get_schema_documentation(), "wrds_compustat")

    # IBES (analyst data)
    with IBESIngestor(WRDS_USER) as ibes:
        s, e = ibes.get_date_range(180)
        recs = ibes.fetch_analyst_recommendations(s, e)
        ests = ibes.fetch_earnings_estimates(s, e)
        ibes.save_data(recs, "wrds/ibes_recommendations")
        ibes.save_data(ests, "wrds/ibes_estimates")
        ibes.save_schema(ibes.get_schema_documentation(), "wrds_ibes")

    # SDC (merge & adquisition events)
    with SDCIngestor(WRDS_USER) as sdc:
        s, e = sdc.get_date_range(180)
        df = sdc.fetch_mna_deals(s, e)
        sdc.save_data(df, "wrds/sdc_mna")
        sdc.save_schema(sdc.get_schema_documentation(), "wrds_sdc")

    # BoardEx (governance)
    with BoardExIngestor(WRDS_USER) as bx:
        s, e = bx.get_date_range(180)
        df = bx.fetch_board_changes(s, e)
        bx.save_data(df, "wrds/boardex_changes")
        bx.save_schema(bx.get_schema_documentation(), "wrds_boardex")

    # FRED (macro)
    fred = FREDIngestor(FRED_KEY)
    s, e = CRSPIngestor(WRDS_USER).get_date_range(180)
    macro = fred.fetch_indicators(["VIXCLS", "FEDFUNDS", "SP500"], s, e)
    fred.save_data(macro, "external/fred_macro")

    # Yahoo Finance (index validation)
    yahoo = YahooFinanceIngestor()
    yf_df = yahoo.fetch_prices(["^GSPC", "^VIX"], s, e)
    yahoo.save_data(yf_df, "external/yahoo_indices")

    # Google Trends (investor attention)
    google = GoogleTrendsIngestor()
    trends = google.fetch_interest(["Nvidia", "Apple", "Tesla"], s, e)
    google.save_data(trends, "external/google_trends")

    logger.info("Ingestion completed in %s seconds", (datetime.now() - start_time).seconds)

if __name__ == "__main__":
    main()
