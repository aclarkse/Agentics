from ._wrds_base import WRDSDataIngestor
import pandas as pd


class CRSPIngestor(WRDSDataIngestor):
    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_daily(self, start: str, end: str) -> pd.DataFrame:
        q = f"""
            SELECT
                -- core stock data
                a.permno, -- security ID
                a. permco, -- company ID
                a.date, -- time key
                a.prc, -- price
                a.shrout, -- shares outstanding
                a.ret, -- return
                a.shrcd, -- share code for filtering
                a.hexcd, -- exchange code for filtering
                
                -- to get the final corrected return
                COALESCE(b.dlretx, a.ret) AS final_ret,
                
                -- compustat link keys
                c.gvkey,
                c.linktype,
                c.linkdt,
                c.linkenddt

            FROM
                -- primary monthly stock data
                crsp.msf AS a

                -- filter by delisting events
            LEFT JOIN
                crsp.dsedelist AS b
                ON a.permno = b.permno
                AND a.date = b.dlstdt

                -- add accounting data from compustat
            LEFT JOIN
                crsp.ccmxpf_linktable AS c
                ON a.permno = c.lpermno
                AND a.date >= c.linkdt
                AND (a.date <= c.linkenddt OR c.linkenddt IS NULL)
                
                -- filter major exchanges and by desired dates
            WHERE
                a.hexcd IN (1, 2, 3)
                AND a.shrcd IN (10, 11)
                AND a.date BETWEEN '{start}' AND '{end}'
        """

        return self.conn.raw_sql(q)

    def fetch_daily_stock_data(self, start: str, end: str) -> pd.DataFrame:
        return self.fetch_daily(start, end)

    @staticmethod
    def get_schema_documentation():
        return {
            "dataset":"CRSP Daily Stock File",
            "library":"crsp",
            "primary_table":"msf",
            "date_field":"date",
            "identifier_fields":["permno","ticker","cusip"]
        }
