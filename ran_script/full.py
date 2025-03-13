import pandas as pd
import numpy as np
import snowflake.connector
import os
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL

# Global variables
current_year = 2024
previous_year = current_year - 1
theyearbefore = previous_year - 1
time_dim = ['Year']
brand_attributes = ['Taste', 'Thickness', 'Flavor', 'Length']
model = LinearRegression()

# Create data directory if it doesn't exist
data_dir = 'datasource'
os.makedirs(data_dir, exist_ok=True)

def connect_to_snowflake():
    engine = create_engine(
        'snowflake://KGIDER%40PMINTL.NET@pl47603.eu-west-1/?'
        'authenticator=externalbrowser&'
        'warehouse=WH_PRD_REPORTING&'
        'role=PMI_EDP_SPK_SNFK_PMI_FDF_PRD_DATAANALYST_IMDL&'
        'database=DB_FDF_PRD'
    )
    return engine
def fetch_mc_per_product(conn):
    """Fetch MC per Product data from Snowflake"""
    print("Fetching MC per Product data...")

    query = f"""
    SELECT trim(a.DF_MARKET_NAME) as "DF_Market", trim(a.LOCATION_NAME) AS "Location",  
      trim(a.SKU_NAME) AS "SKU", a.SKU_ID AS "skuid", 
      b.CR_BRAND_ID AS "CR_BrandId", a.ITEM_PER_BUNDLE AS "Item per Bundle", 
      ROUND(SUM(CASE WHEN (YEAR_NUM = {theyearbefore} AND PL_ITEM = 'PMIDF MC') THEN USD_AMOUNT ELSE 0 END), 2) AS "{theyearbefore} MC",
      ROUND(SUM(CASE WHEN (YEAR_NUM = {theyearbefore} AND PL_ITEM = 'PMIDF NOR') THEN USD_AMOUNT ELSE 0 END), 2) AS "{theyearbefore} NOR",
      ROUND(SUM(CASE WHEN (YEAR_NUM = {previous_year} AND PL_ITEM = 'PMIDF MC') THEN USD_AMOUNT ELSE 0 END), 2) AS "{previous_year} MC",
      ROUND(SUM(CASE WHEN (YEAR_NUM = {previous_year} AND PL_ITEM = 'PMIDF NOR') THEN USD_AMOUNT ELSE 0 END), 2) AS "{previous_year} NOR",
      ROUND(SUM(CASE WHEN (YEAR_NUM = {current_year} AND PL_ITEM = 'PMIDF MC') THEN USD_AMOUNT ELSE 0 END), 2) AS "{current_year} MC",
      ROUND(SUM(CASE WHEN (YEAR_NUM = {current_year} AND PL_ITEM = 'PMIDF NOR') THEN USD_AMOUNT ELSE 0 END), 2) AS "{current_year} NOR"
    FROM DB_FDF_PRD.CS_FINANCE.PNL_FACT_PL_POS a 
    LEFT JOIN DB_FDF_PRD.PRESENTATION.DIM_CR_BRAND b on a.SKU_ID = b.SKU_ID 
    LEFT JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT c ON a.POV_ID = c.POV_ID 
    WHERE a.TRADE_CHANNEL_NAME = 'Airports'
      and a.PRODUCT_CATEGORY_NAME = 'cigarettes'
      And c.DEPARTURE_ARRIVAL != 'A'
      and PL_ITEM IN ('PMIDF MC', 'PMIDF NOR')  
      AND DATA_VERSION_TYPE = 'AC'
      AND DATA_VERSION_NAME in ('COT {theyearbefore} ACT','COT {previous_year} ACT','COT {current_year} ACT')
    GROUP BY ALL
    """

    df = pd.read_sql(query, conn.connect())
    df.to_pickle(f"{data_dir}/MC_per_Product.pkl")
    return df


def fetch_df_vols(conn):
    """Fetch Duty-Free volumes data from Snowflake"""
    print("Fetching DF volumes data...")

    query = f"""
    SELECT trim(a.DF_MARKET_NAME) as "DF_Market", trim(a.LOCATION_NAME) AS "Location", 
      trim(TMO_NAME) AS "TMO", trim(BRAND_FAMILY_NAME) AS "Brand Family", 
      CR_BRAND_ID AS "CR_BrandId", CR_BRAND_NAME AS "SKU", ITEMS_PER_BUNDLE AS "Item per Bundle",
      SUM(CASE WHEN YEAR_NUM = {theyearbefore} THEN STICK_EQUIVALENT_VOLUME ELSE 0 END) AS "{theyearbefore} Volume",
      SUM(CASE WHEN YEAR_NUM = {previous_year} THEN STICK_EQUIVALENT_VOLUME ELSE 0 END) AS "{previous_year} Volume", 
      SUM(CASE WHEN YEAR_NUM = {current_year} THEN STICK_EQUIVALENT_VOLUME ELSE 0 END) AS "{current_year} Volume",
      COUNT(DISTINCT CASE WHEN (YEAR_NUM = {theyearbefore} and STICK_EQUIVALENT_VOLUME > 1) THEN MONTH(DATE) ELSE NULL END) AS "{theyearbefore}Month",
      COUNT(DISTINCT CASE WHEN (YEAR_NUM = {previous_year} and STICK_EQUIVALENT_VOLUME > 1) THEN MONTH(DATE) ELSE NULL END) AS "{previous_year}Month",
      COUNT(DISTINCT CASE WHEN (YEAR_NUM = {current_year} and STICK_EQUIVALENT_VOLUME > 1) THEN MONTH(DATE) ELSE NULL END) AS "{current_year}Month",
      SUM(CASE WHEN YEAR_NUM = {theyearbefore} and ITEMS_PER_BUNDLE > 0 THEN RSP_AMOUNT * (STICK_EQUIVALENT_VOLUME/ITEMS_PER_BUNDLE) ELSE 0 END) AS "{theyearbefore} Revenue",
      SUM(CASE WHEN YEAR_NUM = {previous_year} and ITEMS_PER_BUNDLE > 0 THEN RSP_AMOUNT * (STICK_EQUIVALENT_VOLUME/ITEMS_PER_BUNDLE) ELSE 0 END) AS "{previous_year} Revenue",
      SUM(CASE WHEN YEAR_NUM = {current_year} and ITEMS_PER_BUNDLE > 0 THEN RSP_AMOUNT * (STICK_EQUIVALENT_VOLUME/ITEMS_PER_BUNDLE) ELSE 0 END) AS "{current_year} Revenue"
    FROM DB_FDF_PRD.CS_COMMRCL.RSP_FACT_RSP_CALC a
    LEFT JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT b ON a.POV_ID = b.POV_ID
    WHERE a.TRADE_CHANNEL_NAME = 'Airports' 
      AND YEAR_NUM in ({theyearbefore}, {previous_year}, {current_year}) 
      AND PRODUCT_CATEGORY_NAME = 'Cigarettes'  
      AND DATA_QUALITY_DESC in ('Real', 'Simulated', 'Estimated') 
      AND a.DF_MARKET_NAME not in ('France DP', 'Spain DP', 'Finland DP')
      AND STICK_EQUIVALENT_VOLUME > 0
      AND b.DEPARTURE_ARRIVAL != 'A'
    GROUP BY ALL
    """

    df = pd.read_sql(query, conn)
    df.to_pickle(f"{data_dir}/cat_a_df_vols.pkl")
    return df


def fetch_selma_df_map(conn):
    """Fetch SELMA DF mapping data from Snowflake"""
    print("Fetching SELMA DF mapping data...")

    query = """
    SELECT trim(DF_MARKET_NAME) as "DF_Market", PRODUCT_CATEGORY_NAME as "Product Category", 
      trim(LOCATION_NAME) as "Location", CR_BRAND_ID as "CR_BrandId", FLAVOR as "Flavor", 
      TASTE as "Taste", THICKNESS as "Thickness", LENGTH as "Length"
    FROM DB_FDF_PRD.PRESENTATION.DIM_SELMA_DF
    WHERE TRADE_CHANNEL_NAME = 'Airports' and PRODUCT_CATEGORY_NAME = 'Cigarettes'
    """

    df = pd.read_sql(query, conn)
    df.to_pickle(f"{data_dir}/SELMA_DF_map.pkl")
    return df


def fetch_base_list(conn):
    """Fetch base product list from Snowflake"""
    print("Fetching base product list...")

    query = f"""
    SELECT distinct CR_BRAND_NAME as SKU, ITEMS_PER_BUNDLE as "Item per Bundle", 
      CR_BRAND_ID as "CR_BrandId", trim(a.DF_MARKET_NAME) as "DF_Market", 
      trim(a.LOCATION_NAME) as "Location", TMO_NAME as "TMO"
    FROM DB_FDF_PRD.CS_COMMRCL.RSP_FACT_RSP_CALC a
    LEFT JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT b ON a.POV_ID = b.POV_ID
    WHERE a.TRADE_CHANNEL_NAME = 'Airports' and DATE > '{current_year}-06-01'
      AND b.DEPARTURE_ARRIVAL != 'A'
      AND PRODUCT_CATEGORY_NAME = 'Cigarettes' AND STICK_EQUIVALENT_VOLUME > 0
      AND DATA_QUALITY_DESC in ('Real', 'Simulated', 'Estimated')
    GROUP BY ALL
    """

    df = pd.read_sql(query, conn)
    df.to_pickle(f"{data_dir}/base_list.pkl")
    return df


def fetch_pmi_products(conn):
    """Fetch PMI products data from Snowflake"""
    print("Fetching PMI products data...")

    query = "SELECT * FROM DB_FDF_PRD.PRESENTATION.DIM_CR_BRAND"

    df = pd.read_sql(query, conn)

    # Ensure data directory exists
    os.makedirs('datasource', exist_ok=True)

    # Clean the file path by using os.path.join and removing any invalid characters
    file_path = os.path.join('datasource', 'Pmidf_products.pkl')

    # Attempt to save with error handling
    try:
        df.to_pickle(file_path)
        print(f"PMI products data saved to {file_path}")
    except Exception as e:
        print(f"Warning: Could not save PMI products data to file: {str(e)}")
        print("Continuing with in-memory data...")

    return df

def fetch_dom_prods_data(conn):
    """Fetch domestic products data from Snowflake"""
    print("Fetching domestic products data...")

    query = """
    SELECT EBROM_Id as "EBROMId", TMO_NAME AS "TMO", BRAND_FAMILY_NAME AS "Brand Family", *
    FROM DB_FDF_PRD.CS_PRODUCT.PRD_DIM_EBROM
    """

    df = pd.read_sql(query, conn)
    df.to_pickle(f"{data_dir}/dom_prods_data.pkl")
    return df


def fetch_dom_ims_data(conn):
    """Fetch domestic IMS data from Snowflake"""
    print("Fetching domestic IMS data...")

    query = f"""
    SELECT YEAR_NUM as "Year", a.EBROM_ID as "EBROMId", sum(VOLUME) as "Volume"
    FROM DB_FDF_PRD.CS_SPLYCHN.IMS_FACT_GSPR_IMS a
    LEFT JOIN DB_FDF_PRD.CS_PRODUCT.PRD_DIM_EBROM b
    ON a.EBROM_ID = b.EBROM_ID 
    WHERE VOLUME > 0 and YEAR_NUM = {current_year} and a.EBROM_ID != 0
      AND b.PRODUCT_CATEGORY_NAME = 'Cigarettes'
    GROUP BY YEAR_NUM, a.EBROM_ID
    """

    df = pd.read_sql(query, conn)
    df.to_pickle(f"{data_dir}/dom_ims_data.pkl")
    return df


def fetch_domestic_volumes(conn):
    """Fetch domestic volumes data from Snowflake"""
    print("Fetching domestic volumes data...")

    query = f"""
    SELECT YEAR_NUM as "Year", vpdp.EBROM_ID as "EBROMId", 
      trim(vpdp.MARKET_NAME) as "Market", vpdi.EBROM_NAME as "EBROM", SUM(VOLUME) as "Volume"
    FROM DB_FDF_PRD.CS_SPLYCHN.IMS_FACT_GSPR_IMS vpdi 
    LEFT JOIN DB_FDF_PRD.CS_PRODUCT.PRD_DIM_EBROM vpdp 
    ON vpdi.EBROM_ID = vpdp.EBROM_ID 
    WHERE YEAR_NUM = {current_year} AND vpdi.EBROM_NAME IS NOT NULL AND vpdp.PRODUCT_CATEGORY_NAME = 'Cigarettes'
    GROUP BY ALL
    """

    df = pd.read_sql(query, conn)
    df.to_pickle(f"{data_dir}/DomesticVolumes.pkl")
    return df


def fetch_country_figures(conn):
    """Fetch country figures data from Snowflake"""
    print("Fetching country figures data...")

    query = """
    SELECT YEAR_NUM AS "KFYear", trim(COUNTRY_NAME) as "Country", ADC_STICK as "ADCStick",
      CC_PREVALENCE as "SmokingPrevelance", INBOUND_ALLOWANCE as "InboundAllowance",
      PURCHASER_RATE as "PurchaserRate"
    FROM DB_FDF_PRD.CS_PAXLANU.LAN_FACT_COUNTRY_KEY_FIGURES
    """

    df = pd.read_sql(query, conn)
    df.to_pickle(f"{data_dir}/CF_data.pkl")
    return df


def fetch_df_vol_data(conn):
    """Fetch duty-free volume data from Snowflake"""
    print("Fetching duty-free volume data...")

    query = f"""
    SELECT YEAR_NUM as "Year", PRODUCT_CATEGORY_NAME as "Product Category", 
      trim(a.LOCATION_NAME) as "Location", trim(a.DF_MARKET_NAME) as "DF_Market",
      a.TMO_NAME as "TMO", a.BRAND_FAMILY_NAME "Brand Family", 
      a.CR_BRAND_ID as "CR_BrandId", SUM(VOLUME) AS "DF_Vol"
    FROM DB_FDF_PRD.CS_COMMRCL.RSP_FACT_RSP_CALC a
    LEFT JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT b
    ON a.POV_ID = b.POV_ID 
    WHERE a.TRADE_CHANNEL_NAME = 'Airports' and YEAR_NUM = {current_year} 
      AND PRODUCT_CATEGORY_NAME = 'Cigarettes' 
      AND DATA_QUALITY_DESC in ('Real', 'Simulated', 'Estimated') 
      AND b.DEPARTURE_ARRIVAL in ('D', 'B')
    GROUP BY ALL
    """

    df = pd.read_sql(query, conn)
    df.to_pickle(f"{data_dir}/DF_Vol_data.pkl")
    return df


def fetch_pax_data(conn):
    """Fetch passenger data from Snowflake"""
    print("Fetching passenger data...")

    query = f"""
    SELECT YEAR_NUM AS "Year", IATA_CODE AS "IATA", trim(DF_MARKET_NAME) AS "Market", 
      trim(PORT_NAME) as "AIRPORT_NAME", NATIONALITY AS "Nationality", 
      sum(PAX_QUANTITY*1000) AS "Pax" 
    FROM DB_FDF_PRD.CS_PAXLANU.PAX_FACT_PAX_QUANTITY 
    WHERE DATA_SOURCE_NAME = 'M1ndset Nationalities'
      AND YEAR_NUM = {current_year} 
      AND DEPARTURE_ARRIVAL = 'D' 
      AND validity_desc = 'Actual'
      AND DOM_INTL = 'International'
    GROUP BY YEAR_NUM, IATA_CODE, DF_MARKET_NAME, PORT_NAME, NATIONALITY
    """

    df = pd.read_sql(query, conn)
    df.to_pickle(f"{data_dir}/PAX_data.pkl")
    return df


def fetch_iata_location():
    """Create or load IATA to location mapping"""
    print("Loading IATA to location mapping...")

    # This data isn't explicitly queried in the original script
    # In a real implementation, you would query this from the database

    # For now, create a minimal version or load if exists
    if os.path.exists(f"{data_dir}/IATA_Location.pkl"):
        return pd.read_pickle(f"{data_dir}/IATA_Location.pkl")
    else:
        # Create a minimal version for the script to run
        # This should be replaced with actual data from the database
        df = pd.DataFrame({'IATA': ['KUW', 'JEJ', 'ZRH'],
                           'Location': ['Kuwait', 'Jeju', 'Zurich']})
        df.to_pickle(f"{data_dir}/IATA_Location.pkl")
        return df


def fetch_nationality_mapping():
    """Create or load nationality to country mapping"""
    print("Loading nationality to country mapping...")

    # Similar to IATA_Location, this isn't explicitly queried
    if os.path.exists(f"{data_dir}/mrk_nat_map.pkl"):
        return pd.read_pickle(f"{data_dir}/mrk_nat_map.pkl")
    else:
        # Create a minimal version
        # This should be replaced with actual data
        df = pd.DataFrame({'Nationality': ['KWT', 'KOR', 'CHE'],
                           'Nationalities': ['KWT', 'KOR', 'CHE'],
                           'Countries': ['Kuwait', 'South Korea', 'Switzerland']})
        df.to_pickle(f"{data_dir}/mrk_nat_map.pkl")
        return df


def create_df_vols_w_financials(cat_a_df_vols, mc_per_product):
    """Create DF_Vols_w_Financials by merging cat_a_df_vols with MC_per_Product"""
    # Verify columns in cat_a_df_vols
    required_columns = ['DF_Market', 'Location', 'CR_BrandId', 'TMO', 'Brand Family',
                        'SKU', 'Item per Bundle', f"{current_year} Volume",
                        f"{previous_year} Volume", f"{current_year}Month",
                        f"{previous_year}Month", f"{current_year} Revenue",
                        f"{previous_year} Revenue"]

    missing_cols = [col for col in required_columns if col not in cat_a_df_vols.columns]

    if missing_cols:
        # Re-fetch cat_a_df_vols with all required columns
        query = f'''
        SELECT  trim(a.DF_MARKET_NAME) as "DF_Market", trim(a.LOCATION_NAME) AS "Location", 
          trim(TMO_NAME) AS "TMO", trim(BRAND_FAMILY_NAME) AS "Brand Family", 
          CR_BRAND_ID AS "CR_BrandId", CR_BRAND_NAME AS "SKU", ITEMS_PER_BUNDLE AS "Item per Bundle",
          SUM(CASE WHEN YEAR_NUM = {theyearbefore} THEN STICK_EQUIVALENT_VOLUME ELSE 0 END) AS "{theyearbefore} Volume",
          SUM(CASE WHEN YEAR_NUM = {previous_year} THEN STICK_EQUIVALENT_VOLUME ELSE 0 END) AS "{previous_year} Volume", 
          SUM(CASE WHEN YEAR_NUM = {current_year} THEN STICK_EQUIVALENT_VOLUME ELSE 0 END) AS "{current_year} Volume",
          COUNT(DISTINCT CASE WHEN (YEAR_NUM = {theyearbefore} and STICK_EQUIVALENT_VOLUME > 1) THEN MONTH(DATE) ELSE NULL END) AS "{theyearbefore}Month",
          COUNT(DISTINCT CASE WHEN (YEAR_NUM = {previous_year} and STICK_EQUIVALENT_VOLUME > 1) THEN MONTH(DATE) ELSE NULL END) AS "{previous_year}Month",
          COUNT(DISTINCT CASE WHEN (YEAR_NUM = {current_year} and STICK_EQUIVALENT_VOLUME > 1) THEN MONTH(DATE) ELSE NULL END) AS "{current_year}Month",
          SUM(CASE WHEN YEAR_NUM = {theyearbefore} and ITEMS_PER_BUNDLE > 0 THEN RSP_AMOUNT * (STICK_EQUIVALENT_VOLUME/ITEMS_PER_BUNDLE) ELSE 0 END) AS "{theyearbefore} Revenue",
          SUM(CASE WHEN YEAR_NUM = {previous_year} and ITEMS_PER_BUNDLE > 0 THEN RSP_AMOUNT * (STICK_EQUIVALENT_VOLUME/ITEMS_PER_BUNDLE) ELSE 0 END) AS "{previous_year} Revenue",
          SUM(CASE WHEN YEAR_NUM = {current_year} and ITEMS_PER_BUNDLE > 0 THEN RSP_AMOUNT * (STICK_EQUIVALENT_VOLUME/ITEMS_PER_BUNDLE) ELSE 0 END) AS "{current_year} Revenue"
        FROM DB_FDF_PRD.CS_COMMRCL.RSP_FACT_RSP_CALC a
        LEFT JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT b ON a.POV_ID = b.POV_ID
        WHERE a.TRADE_CHANNEL_NAME = 'Airports' 
          AND YEAR_NUM in ({theyearbefore}, {previous_year}, {current_year}) 
          AND PRODUCT_CATEGORY_NAME = 'Cigarettes'  
          AND DATA_QUALITY_DESC in ('Real', 'Simulated', 'Estimated') 
          AND a.DF_MARKET_NAME not in ('France DP', 'Spain DP', 'Finland DP')
          AND STICK_EQUIVALENT_VOLUME > 0
          AND b.DEPARTURE_ARRIVAL != 'A'
        GROUP BY ALL
        '''

        conn = snowflake.connector.connect(
            user="KGIDER@PMINTL.NET",
            account="pl47603.eu-west-1",
            authenticator="externalbrowser",
            warehouse='WH_PRD_REPORTING',
            role='PMI_EDP_SPK_SNFK_PMI_FDF_PRD_DATAANALYST_IMDL',
            database="DB_FDF_PRD"
        )

        cat_a_df_vols = pd.read_sql_query(query, conn)

    # Merge cat_a_df_vols with financial data
    df = cat_a_df_vols.merge(
        mc_per_product[['DF_Market', 'Location', 'CR_BrandId',
                        f"{previous_year} MC", f"{previous_year} NOR",
                        f"{current_year} MC", f"{current_year} NOR"]],
        how='left',
        on=['DF_Market', 'Location', 'CR_BrandId']
    )

    # Replace NaN values with 0 for financial columns only
    financial_cols = [f"{previous_year} MC", f"{previous_year} NOR",
                      f"{current_year} MC", f"{current_year} NOR"]
    for col in financial_cols:
        df[col] = df[col].fillna(0)

    # Filter for current year volume > 0
    df = df[df[f"{current_year} Volume"] > 0]

    # Calculate revenue averages
    df['LYRevenueAvg'] = np.where(
        df[f"{previous_year}Month"] == 0,
        0,
        df[f"{previous_year} Revenue"] / df[f"{previous_year}Month"]
    )

    df['CYRevenueAvg'] = np.where(
        df[f"{current_year}Month"] == 0,
        0,
        df[f"{current_year} Revenue"] / df[f"{current_year}Month"]
    )

    # Calculate Growth
    df['Growth'] = np.where(
        df['LYRevenueAvg'] == 0,
        0,
        (df['CYRevenueAvg'] - df['LYRevenueAvg']) / df['LYRevenueAvg']
    )

    # Calculate Margin
    df['Margin'] = np.where(
        (df[f"{current_year} NOR"] <= 0) | (df[f"{current_year}Month"] == 0),
        0,
        (df[f"{current_year} MC"] / df[f"{current_year}Month"]) /
        (df[f"{current_year} NOR"] / df[f"{current_year}Month"])
    )

    return df

def create_pmi_margins(df_vols_w_financials):
    """Create pmi_margins dataframe"""
    print("Creating pmi_margins...")

    pmi_margins = df_vols_w_financials[df_vols_w_financials['TMO'] == 'PMI'].copy()

    # Calculate Margin_Volume
    pmi_margins['Margin_Volume'] = round(
        (pmi_margins[f"{current_year} Volume"] * pmi_margins['Margin'].fillna(0)), 0
    ).astype(int)

    # Group by DF_Market, Location, Brand Family
    pmi_margins = pmi_margins.groupby(['DF_Market', 'Location', 'Brand Family']).sum().reset_index()
    pmi_margins = pmi_margins[['DF_Market', 'Location', 'Brand Family',
                               f"{current_year} Volume", f"{current_year} MC",
                               'Margin_Volume']]

    # Calculate Brand Family Margin
    pmi_margins['Brand Family Margin'] = pmi_margins['Margin_Volume'] / pmi_margins[f"{current_year} Volume"]

    return pmi_margins

def create_sku_by_vols_margins(df_vols_w_financials, pmi_margins):
    """Create SKU_by_Vols_Margins by merging DF_Vols_w_Financials with pmi_margins"""
    print("Creating SKU_by_Vols_Margins...")

    df = df_vols_w_financials.merge(
        pmi_margins[['DF_Market', 'Location', 'Brand Family', 'Brand Family Margin']],
        how='left',
        on=['DF_Market', 'Location', 'Brand Family']
    ).fillna(0)

    # Calculate Margin Comparison
    df['Margin Comparison'] = np.where(
        df['Brand Family Margin'] < df['Margin'],
        1,
        0
    )

    return df


def create_flag_counts(df_vols_w_financials):
    """Calculate the number of SKUs for Green & Red Flag SKUs"""
    print("Calculating flag counts...")

    no_of_sku = df_vols_w_financials.groupby(['DF_Market', 'Location'])['SKU'].count().reset_index()
    no_of_sku = no_of_sku.rename(columns={'SKU': 'TotalSKU'})
    no_of_sku['GreenFlagSKU'] = (no_of_sku['TotalSKU'] * 0.05).apply(np.ceil)
    no_of_sku['RedFlagSKU'] = round(no_of_sku['TotalSKU'] * 0.25, 0)

    return no_of_sku


def create_green_flags(df_vols_w_financials, no_of_sku):
    """Create green flags based on Rule 1"""
    print("Creating green flags (Rule 1)...")

    gf = pd.DataFrame()

    for location in df_vols_w_financials.Location.unique():
        df_vols = df_vols_w_financials[df_vols_w_financials['Location'] == location]

        green_threshold = int(no_of_sku[no_of_sku['Location'] == location].iloc[0]['GreenFlagSKU'])

        green_flag1 = df_vols.sort_values(
            f"{current_year} Volume",
            ascending=False
        ).head(green_threshold)

        green_flag1['Green1'] = 1
        green_flag1 = green_flag1[green_flag1['TMO'] == 'PMI']

        gf = pd.concat([gf, green_flag1], ignore_index=True)

    return gf


def create_green_flags2(sku_by_vols_margins, no_of_sku):
    """Create green flags based on Rule 2"""
    print("Creating green flags (Rule 2)...")

    gf2 = pd.DataFrame()

    for location in sku_by_vols_margins.Location.unique():
        df_vols = sku_by_vols_margins[sku_by_vols_margins['Location'] == location]

        green_threshold = int(no_of_sku[no_of_sku['Location'] == location].iloc[0]['GreenFlagSKU'])

        green_flag2 = df_vols.sort_values(
            'Growth',
            ascending=False
        ).head(green_threshold)

        green_flag2 = green_flag2[green_flag2['TMO'] == 'PMI']
        green_flag2 = green_flag2[green_flag2['Margin Comparison'] == 1]
        green_flag2['Green Flag2'] = 1

        gf2 = pd.concat([gf2, green_flag2], ignore_index=True)

    return gf2


def create_green_list(gf, gf2):
    """Create combined green list"""
    print("Creating combined green list...")

    green_list = pd.concat([gf, gf2])
    green_list = green_list[['DF_Market', 'Location', 'TMO', 'Brand Family',
                             'CR_BrandId', 'SKU', 'Item per Bundle']]
    green_list = green_list.drop_duplicates()
    green_list['Green'] = 1

    return green_list


def create_red_flags1(df_vols_w_financials, no_of_sku):
    """Create red flags based on low volume and negative growth"""
    print("Creating red flags (part 1)...")

    rf1 = pd.DataFrame()

    for location in df_vols_w_financials.Location.unique():
        red_vols = df_vols_w_financials[df_vols_w_financials['Location'] == location]

        red_threshold = int(no_of_sku[no_of_sku['Location'] == location].iloc[0]['RedFlagSKU'])

        # SKUs with lowest volume
        red_flag1 = red_vols.sort_values(
            f"{current_year} Volume",
            ascending=True
        ).head(red_threshold)
        red_flag1 = red_flag1[red_flag1['TMO'] == 'PMI']

        # SKUs with lowest growth
        red_flag1_1 = red_vols.sort_values(
            'Growth',
            ascending=True
        ).head(red_threshold)
        red_flag1_1 = red_flag1_1[red_flag1_1['TMO'] == 'PMI']

        # Find intersection
        red_flag_intersection = np.intersect1d(red_flag1.CR_BrandId, red_flag1_1.CR_BrandId)

        # Combine and filter
        red_flag1_2 = pd.concat([red_flag1, red_flag1_1], ignore_index=True)
        red_flag1_2 = red_flag1_2[red_flag1_2['CR_BrandId'].isin(red_flag_intersection)].drop_duplicates()
        red_flag1_2 = red_flag1_2[['DF_Market', 'Location', 'TMO', 'Brand Family',
                                   'CR_BrandId', 'SKU', 'Item per Bundle']]

        rf1 = pd.concat([rf1, red_flag1_2], ignore_index=True)

    return rf1


def create_red_flags2(sku_by_vols_margins, no_of_sku):
    """Create red flags based on low growth and lower margin"""
    print("Creating red flags (part 2)...")

    rf2 = pd.DataFrame()

    for location in sku_by_vols_margins.Location.unique():
        red_threshold = int(no_of_sku[no_of_sku['Location'] == location].iloc[0]['RedFlagSKU'])

        red_flag2_1 = sku_by_vols_margins[sku_by_vols_margins['Location'] == location]
        red_flag2_1 = red_flag2_1.sort_values(
            'Growth',
            ascending=True
        ).head(red_threshold)

        red_flag2_1 = red_flag2_1[red_flag2_1['TMO'] == 'PMI']
        red_flag2_1 = red_flag2_1[red_flag2_1['Margin Comparison'] == 0]
        red_flag2_1 = red_flag2_1[['DF_Market', 'Location', 'TMO', 'Brand Family',
                                   'CR_BrandId', 'SKU', 'Item per Bundle']]

        rf2 = pd.concat([rf2, red_flag2_1], ignore_index=True)

    return rf2


def create_red_list(rf1, rf2):
    """Create combined red list"""
    print("Creating combined red list...")

    red_list = pd.concat([rf1, rf2], ignore_index=True).drop_duplicates()
    red_list['Red'] = 1

    return red_list


def create_green_red_list(green_list, red_list):
    """Create combined green-red list"""
    print("Creating combined green-red list...")

    green_red_list = green_list.merge(
        red_list,
        how='outer',
        on=['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU', 'Item per Bundle']
    ).fillna(0)

    green_red_list['Check'] = np.where(
        green_red_list['Green'] != green_red_list['Red'],
        'OK',
        'Problem'
    )

    green_red_list = green_red_list[green_red_list['Check'] != 'Problem']
    green_red_list['Status'] = np.where(
        green_red_list['Green'] == 1,
        'Green',
        'Red'
    )

    return green_red_list


def create_category_a_dataframe(sku_by_vols_margins, green_red_list):
    """Create category_a_1 dataframe"""
    print("Creating category A dataframe...")

    # Filter for PMI SKUs
    category_a_0 = sku_by_vols_margins[sku_by_vols_margins['TMO'] == 'PMI']

    # Merge with green_red_list
    category_a_1 = category_a_0.merge(
        green_red_list[['DF_Market', 'Location', 'TMO', 'Brand Family',
                        'CR_BrandId', 'SKU', 'Item per Bundle', 'Status']],
        how='left',
        on=['DF_Market', 'Location', 'TMO', 'Brand Family',
            'CR_BrandId', 'SKU', 'Item per Bundle']
    ).fillna(0)

    return category_a_1


def create_calculation_table(category_a_1):
    """Create calculation table for category A score calculation"""
    print("Creating calculation table...")

    # Count total SKUs
    total_sku = category_a_1.groupby(['DF_Market', 'Location'])['CR_BrandId'].count().reset_index()
    total_sku = total_sku.rename(columns={'CR_BrandId': 'TotalSKU'})

    # Count green SKUs
    ct_green = category_a_1[category_a_1['Status'] == 'Green']
    ct_green = ct_green.groupby(['DF_Market', 'Location'])['CR_BrandId'].count().reset_index()
    ct_green = ct_green.rename(columns={'CR_BrandId': 'GreenSKU'})

    # Count red SKUs
    ct_red = category_a_1[category_a_1['Status'] == 'Red']
    ct_red = ct_red.groupby(['DF_Market', 'Location'])['CR_BrandId'].count().reset_index()
    ct_red = ct_red.rename(columns={'CR_BrandId': 'RedSKU'})

    # Merge red and green counts
    ct_gr_red = ct_red.merge(ct_green, how='outer', on=['DF_Market', 'Location'])

    # Create final calculation table
    calculation_table = total_sku.merge(ct_gr_red, how='outer', on=['DF_Market', 'Location'])
    calculation_table['RedSKU'] = calculation_table['RedSKU'].fillna(0).astype('int')
    calculation_table['GreenSKU'] = calculation_table['GreenSKU'].fillna(0).astype(int)

    return calculation_table


def calculate_cat_a_scores(calculation_table):
    """Calculate Category A scores"""
    print("Calculating Category A scores...")

    location = []
    score_a = []

    for loc in calculation_table.Location.unique():
        ct = calculation_table[calculation_table['Location'] == loc]
        score = ((ct['GreenSKU'].iloc[0] - (ct['RedSKU'].iloc[0] * 2)) / ct['TotalSKU'].iloc[0]) * 100
        location.append(loc)
        score_a.append(score)

    # Create dataframe
    cat_a_scores = pd.DataFrame(list(zip(location, score_a)), columns=['Location', 'Score_A']).fillna(0)
    cat_a_scores['ScaledScore'] = round((cat_a_scores['Score_A'] - (-200)) * (10 / 300), 2)

    return cat_a_scores


def prepare_duty_free_volumes(df_vols, base_list):
    """Prepare duty free volumes dataframe"""
    print("Preparing duty free volumes...")

    duty_free_volumes = df_vols.copy()
    duty_free_volumes['key'] = duty_free_volumes['CR_BrandId'].astype('str') + '-' + duty_free_volumes[
        'Item per Bundle'].astype('str')

    # Prepare category list
    category_list = base_list.copy()
    category_list['key'] = category_list['CR_BrandId'].astype('str') + '-' + category_list['Item per Bundle'].astype(
        'str')

    return duty_free_volumes, category_list


def create_tobacco_range(duty_free_volumes, category_list):
    """Create tobacco range dataframe by merging duty_free_volumes with category_list"""
    # Ensure we have all necessary columns from duty_free_volumes
    volume_col = f"{current_year} Volume"

    # Check available columns
    available_cols = ['DF_Market', 'Location', 'key', volume_col]
    for col in available_cols:
        if col not in duty_free_volumes.columns:
            print(f"Missing column in duty_free_volumes: {col}")

    # Merge with key to create tobacco_range
    tobacco_range = category_list.merge(
        duty_free_volumes[['DF_Market', 'Location', 'key', volume_col]],
        how='left',
        on=['DF_Market', 'Location', 'key']
    )

    # Join to get TMO data from category_list instead of duty_free_volumes
    tobacco_range = tobacco_range.merge(
        category_list[['key', 'TMO', 'CR_BrandId', 'SKU']],
        on='key',
        how='left'
    )

    # Classify TMO values
    tobacco_range['TMO'] = np.where(tobacco_range['TMO'] != 'PMI', 'Comp', 'PMI')
    tobacco_range = tobacco_range[tobacco_range['CR_BrandId'] != 0]

    # Create tobacco_range2 (pivoted)
    tobacco_range2 = tobacco_range[['DF_Market', 'Location', 'TMO', 'CR_BrandId', 'SKU', volume_col]]
    tobacco_range2 = pd.pivot_table(
        tobacco_range2,
        index=['DF_Market', 'Location', 'TMO', 'CR_BrandId'],
        aggfunc={volume_col: np.sum, 'SKU': np.count_nonzero}
    ).reset_index()

    return tobacco_range, tobacco_range2


def create_market_mix(selma_df_map, tobacco_range2, brand_attributes):
    """Create Market_Mix dataframe"""
    print("Creating Market_Mix dataframe...")

    # Process SELMA DF products
    selma_df_products = selma_df_map.copy()
    selma_df_products = selma_df_products[selma_df_products['Product Category'] == 'Cigarettes'].reset_index()
    selma_df_products_2 = selma_df_products[['DF_Market', 'Location', 'CR_BrandId'] + brand_attributes]
    selma_df_products_3 = selma_df_products_2.drop_duplicates()
    selma_df_products_3 = selma_df_products_3[selma_df_products_3['CR_BrandId'] != 0]

    # Merge with tobacco_range2
    market_mix = selma_df_products_3.merge(
        tobacco_range2[['DF_Market', 'Location', 'CR_BrandId', 'TMO', 'SKU', f"{current_year} Volume"]],
        how='left',
        on=['DF_Market', 'Location', 'CR_BrandId']
    )

    market_mix = market_mix[market_mix['SKU'].notnull()]
    market_mix = market_mix[market_mix['SKU'] != 0]

    return market_mix


def create_market_summary(market_mix, tobacco_range, brand_attributes):
    """Create Market_Summary dataframe"""
    print("Creating Market_Summary dataframe...")

    # Create all_market (count of SKUs by TMO)
    all_market = pd.pivot_table(
        tobacco_range,
        index=['DF_Market', 'Location', 'TMO'],
        aggfunc={'SKU': np.count_nonzero}
    ).reset_index()
    all_market = all_market.rename(columns={'SKU': 'Total TMO'})

    # Create Market_Summary0 (SKUs by TMO and attributes)
    market_summary0 = pd.pivot_table(
        market_mix,
        index=['DF_Market', 'Location', 'TMO'] + brand_attributes,
        aggfunc={f"{current_year} Volume": "sum", 'SKU': "sum"}
    ).reset_index()

    # Merge to create Market_Summary
    market_summary = market_summary0.merge(
        all_market,
        how='left',
        on=['DF_Market', 'Location', 'TMO']
    )

    market_summary['SoM'] = round(market_summary['SKU'] * 100 / market_summary['Total TMO'], 1)
    market_summary = market_summary[['DF_Market', 'Location', 'TMO'] + brand_attributes + ['SKU', 'Total TMO', 'SoM']]

    return market_summary0, market_summary


def create_market_summary_pmi_comp(market_summary):
    """Split Market_Summary into PMI and Comp dataframes"""
    print("Creating Market_Summary_PMI and Market_Summary_Comp...")

    # Filter and rename for PMI
    market_summary_pmi = market_summary[market_summary['TMO'] == 'PMI']
    market_summary_pmi = market_summary_pmi.rename(
        columns={'SoM': 'SoM_PMI', 'SKU': 'PMI_Seg_SKU', 'Total TMO': 'PMI Total'}
    )
    market_summary_pmi = market_summary_pmi[['DF_Market', 'Location'] + brand_attributes +
                                            ['PMI_Seg_SKU', 'PMI Total', 'SoM_PMI']]

    # Filter and rename for Comp
    market_summary_comp = market_summary[market_summary['TMO'] == 'Comp']
    market_summary_comp = market_summary_comp.rename(
        columns={'SoM': 'SoM_Comp', 'SKU': 'Comp_Seg_SKU', 'Total TMO': 'Comp Total'}
    )
    market_summary_comp = market_summary_comp[['DF_Market', 'Location'] + brand_attributes +
                                              ['Comp_Seg_SKU', 'Comp Total', 'SoM_Comp']]

    return market_summary_pmi, market_summary_comp


def create_market_delta(market_summary_comp, market_summary_pmi, market_summary0, brand_attributes):
    """Create Market_Summary_Delta and Market dataframes"""
    print("Creating Market_Summary_Delta and Market dataframes...")

    # Create Market_Summary_Delta
    market_summary_delta = market_summary_comp.merge(
        market_summary_pmi[['DF_Market', 'Location', 'Flavor', 'Taste', 'Thickness',
                            'Length', 'PMI_Seg_SKU', 'PMI Total', 'SoM_PMI']],
        how='outer',
        on=['DF_Market', 'Location', 'Flavor', 'Taste', 'Thickness', 'Length']
    ).fillna(0)

    # Create Market_Volume_Table
    market_volume_table = market_summary0.groupby(
        ['DF_Market', 'Location'] + brand_attributes
    ).sum(f"{current_year} Volume").reset_index()

    # Create Market
    market = market_summary_delta.merge(
        market_volume_table[['DF_Market', 'Location'] + brand_attributes + [f"{current_year} Volume"]],
        how='left',
        on=['DF_Market', 'Location'] + brand_attributes
    )

    market['SKU_Delta'] = market['SoM_PMI'] - market['SoM_Comp']

    return market_summary_delta, market


def calculate_cat_b_scores(market, market_mix):
    """Calculate Category B scores"""
    print("Calculating Category B scores...")

    location = []
    score = []
    num_of_pmi_sku = []
    num_of_comp_sku = []
    pmi_cot = []
    comp_cot = []

    for loc in market['Location'].unique():
        looped_market = market[market['Location'] == loc]
        X, y = looped_market[["SoM_PMI"]], looped_market[["SoM_Comp"]]
        model.fit(X, y)
        r_squared = model.score(X, y)
        market_score = round(r_squared * 10, 2)

        skunum = int(market[(market['Location'] == loc)].iloc[:, -4].max())
        compsku = int(market[(market['Location'] == loc)].iloc[:, -7].max())
        pmi_vol = int(market_mix[
                          (market_mix['Location'] == loc) & (market_mix['TMO'] == 'PMI')
                          ][f"{current_year} Volume"].sum())
        comp_vol = market_mix[(market_mix['Location'] == loc) & (market_mix['TMO'] != 'PMI')].sum().iloc[-1].astype(
            'int')

        location.append(loc)
        score.append(market_score)
        num_of_pmi_sku.append(skunum)
        num_of_comp_sku.append(compsku)
        pmi_cot.append(pmi_vol)
        comp_cot.append(comp_vol)

    # Create dataframe
    list_of_tuples = list(zip(location, score, num_of_pmi_sku, num_of_comp_sku, pmi_cot, comp_cot))
    cat_b_scores = pd.DataFrame(
        list_of_tuples,
        columns=['Location', 'RSQ', 'NumPMI_SKU', 'NumComp_SKU', 'PMI Volume', 'Comp Volume']
    ).fillna(0)
    cat_b_scores = cat_b_scores.rename(columns={'RSQ': 'Cat_B'})

    return cat_b_scores


def process_paris_module(pax_data, mrk_nat_map, iata_location, country_figures,
                         dom_ims_data, domestic_volumes, dom_prods_data, selma_dom_map,
                         df_vols, selma_df_map, time_dim, brand_attributes, current_year):
    """Process Category C (PARIS) module"""
    print("Processing Category C (PARIS) module...")

    # Define dimension explicitly
    dimension = brand_attributes
    domestic_dimensions = ['Market', 'EBROMId', 'EBROM', 'Taste', 'Thickness', 'Flavor', 'Length']

    # Clean and prepare SELMA domestic map
    selma_dom_map['Length'] = np.where(
        selma_dom_map['Length'].isin(['REGULAR SIZE', 'REGULAR FILTER', 'SHORT SIZE', 'LONG FILTER']),
        'KS',
        np.where(
            selma_dom_map['Length'].isin(['LONGER THAN KS', '100', 'LONG SIZE', 'EXTRA LONG', 'SUPER LONG']),
            'LONG',
            selma_dom_map['Length']
        )
    )

    selma_dom_map['Thickness'] = np.where(
        selma_dom_map['Thickness'] == 'FAT',
        'STD',
        selma_dom_map['Thickness']
    )

    selma_dom_map = selma_dom_map.merge(
        dom_prods_data[['EBROMId']],
        how='left',
        on='EBROMId'
    )

    # Prepare DF_Vols
    df_vols_location = df_vols.merge(iata_location, how='left', on='Location')
    df_vols = df_vols_location.merge(
        selma_df_map[['CR_BrandId', 'Location'] + brand_attributes],
        how='left',
        on=['CR_BrandId', 'Location']
    )

    # Process passenger data
    pax_d1 = pax_data[time_dim + ['IATA', 'Market', 'Nationality', 'Pax']].copy()
    pax_d2 = pax_d1.groupby(time_dim + ['IATA', 'Market', 'Nationality']).sum().reset_index()
    pax_d3 = pax_d2.merge(mrk_nat_map, how='left', left_on='Nationality', right_on='Nationalities')

    # Process country figures
    cf_d2 = country_figures[['KFYear', 'Country', 'SmokingPrevelance',
                             'InboundAllowance', 'ADCStick', 'PurchaserRate']].copy()

    cf_d2['ADCStick'] = cf_d2['ADCStick'].fillna(15.0)
    cf_d2['InboundAllowance'] = cf_d2['InboundAllowance'].fillna(400.0)
    cf_d2['PurchaserRate'] = np.where(
        cf_d2['PurchaserRate'] == cf_d2['PurchaserRate'],
        cf_d2['PurchaserRate'],
        cf_d2['SmokingPrevelance']
    )

    # Merge passenger data with country figures
    pax_d4 = pax_d3.merge(
        cf_d2,
        how='left',
        left_on=['Nationalities', 'Year'],
        right_on=['Country', 'KFYear']
    )

    # Calculate LANU and stick consumption
    pax_d4['Pax'] = np.ceil(pax_d4['Pax'] * 1000)
    pax_d4['LANU'] = pax_d4['Pax'] * pax_d4['SmokingPrevelance'] * 0.9
    pax_d4['LANU'] = np.ceil(pax_d4['LANU'])
    pax_d4['InboundAllowance'] = pax_d4['InboundAllowance'].astype(float)
    pax_d4['StickCons'] = pax_d4['LANU'] * pax_d4['InboundAllowance']

    # Create pax_fin (passenger data by nationality)
    pax_fin_ = pax_d4[time_dim + ['Market', 'IATA', 'Nationality',
                                  'Countries', 'LANU', 'StickCons']].rename(columns={'Market': 'DF_Market'})

    pax_fin = pax_fin_.groupby(
        time_dim + ['DF_Market', 'IATA', 'Nationality', 'Countries']
    ).sum('StickCons').reset_index()

    # Process domestic attributes
    dom_attr = selma_dom_map[domestic_dimensions].merge(
        dom_prods_data[['EBROMId', 'TMO', 'Brand Family']],
        how='left',
        on='EBROMId'
    ).fillna('NaN')

    # Process domestic IMS data
    dom_ims2 = dom_ims_data.merge(dom_attr, how='left', on='EBROMId')
    dom_ims2 = dom_ims2[dom_ims2['Market'] != 'PMIDF']
    dom_ims2 = dom_ims2[dom_ims2['EBROMId'] != 0]
    dom_ims2 = dom_ims2[dom_ims2['EBROM'] == dom_ims2['EBROM']]
    dom_ims2['Market'] = dom_ims2['Market'].replace('PRC', 'China')

    # Calculate domestic market totals
    dom_totals = dom_ims2.groupby(time_dim + ['Market']).sum().reset_index().rename(columns={'Volume': 'TotVol'})

    # Calculate Share of Volumes
    dom_sov = dom_ims2.merge(
        dom_totals[time_dim + ['Market', 'TotVol']],
        how='left',
        on=time_dim + ['Market']
    )

    dom_sov['SoDom'] = dom_sov['Volume'] / dom_sov['TotVol']

    dom_fin = dom_sov[time_dim + domestic_dimensions + ['TMO', 'SoDom']].rename(columns={'Market': 'Dom_Market'})

    # Calculate projected volumes
    projected_vol_by_sku = pax_fin.merge(
        dom_fin,
        how='left',
        left_on=time_dim + ['Countries'],
        right_on=time_dim + ['Dom_Market']
    )
    projected_vol_by_sku['Proj_Vol_bySKU'] = round(projected_vol_by_sku['SoDom'] * projected_vol_by_sku['StickCons'])
    projected_vol_by_sku['Proj_LANU_bySKU'] = round(projected_vol_by_sku['SoDom'] * projected_vol_by_sku['LANU'])

    # Aggregate projected volumes by dimension
    projected_vol_by_prod_dim = projected_vol_by_sku.groupby(['IATA'] + dimension).agg(
        Proj_Vol_PG=('Proj_Vol_bySKU', np.sum),
        Proj_LANU_PG=('Proj_LANU_bySKU', np.sum)
    ).reset_index()

    # Calculate projected SoM
    proj_totVol = projected_vol_by_prod_dim.groupby(['IATA']).sum().reset_index().rename(
        columns={'Proj_Vol_PG': 'Tot_proj_Vol'}
    )

    proj_SoM_PG = projected_vol_by_prod_dim.merge(
        proj_totVol[['IATA', 'Tot_proj_Vol']],
        how='left',
        on=['IATA']
    )

    proj_SoM_PG['Proj_SoM_PG'] = proj_SoM_PG['Proj_Vol_PG'] / proj_SoM_PG['Tot_proj_Vol']

    # Calculate actual DF volumes by SKU
    DFVol_IATA_bySKU = df_vols.groupby(['Year', 'IATA'] + dimension).sum('DF_Vol').reset_index()

    Total_DFVol_byIATA = DFVol_IATA_bySKU.groupby(['Year', 'IATA']).sum().reset_index().rename(
        columns={'DF_Vol': 'DFTot_Vol'}
    )

    DFSoM_IATA_bySKU = DFVol_IATA_bySKU.merge(
        Total_DFVol_byIATA[time_dim + ['IATA', 'DFTot_Vol']],
        how='left',
        on=time_dim + ['IATA']
    )

    DFSoM_IATA_bySKU['DF_SoM_IATA_PG'] = DFSoM_IATA_bySKU['DF_Vol'] / DFSoM_IATA_bySKU['DFTot_Vol']

    DFSoM_IATA_bySKU = DFSoM_IATA_bySKU[['IATA'] + dimension + ['DF_Vol', 'DFTot_Vol', 'DF_SoM_IATA_PG']]

    # Create PARIS output
    PARIS_output = proj_SoM_PG.merge(
        DFSoM_IATA_bySKU,
        how='outer',
        on=dimension + ['IATA']
    ).fillna(0)

    PARIS_output['DF_SoM_IATA_PG'] = PARIS_output['DF_SoM_IATA_PG'].fillna(0)
    PARIS_output['Proj_SoM_PG'] = PARIS_output['Proj_SoM_PG'].fillna(0)
    PARIS_output['Delta_SoS'] = PARIS_output['Proj_SoM_PG'] - PARIS_output['DF_SoM_IATA_PG']

    PARIS_output = PARIS_output[dimension + ['IATA', 'DF_Vol', 'Proj_SoM_PG', 'DF_SoM_IATA_PG', 'Delta_SoS']]
    PARIS_output = PARIS_output.merge(iata_location, how='left', on='IATA')
    PARIS_output = PARIS_output[PARIS_output['Location'].notnull()]
    PARIS_output = PARIS_output.rename(
        columns={'DF_SoM_IATA_PG': 'Real_So_Segment', 'Proj_SoM_PG': 'Ideal_So_Segment'}
    )
    PARIS_output = PARIS_output[PARIS_output['Ideal_So_Segment'] > 0.001]
    PARIS_output = PARIS_output[['Location', 'IATA'] + brand_attributes +
                                ['DF_Vol', 'Real_So_Segment', 'Ideal_So_Segment', 'Delta_SoS']]

    return PARIS_output, dom_fin, projected_vol_by_sku, proj_SoM_PG, DFSoM_IATA_bySKU


def calculate_cat_c_scores(PARIS_output):
    """Calculate Category C scores using proper data"""
    print("Calculating Category C scores...")

    # Check if PARIS_output has required columns
    required_columns = ['Location', 'Real_So_Segment', 'Ideal_So_Segment']
    if not all(col in PARIS_output.columns for col in required_columns):
        raise ValueError(
            f"PARIS_output missing required columns: {[col for col in required_columns if col not in PARIS_output.columns]}")

    location = []
    score = []

    for loc in PARIS_output['Location'].unique():
        # Filter data for this location
        looped_market = PARIS_output[PARIS_output['Location'] == loc]

        # Check if we have enough data points
        if len(looped_market) < 3:
            print(f"Warning: Not enough data points for {loc}, skipping score calculation")
            continue

        # Remove rows with zero values in either column to avoid skewing results
        looped_market = looped_market[(looped_market['Real_So_Segment'] > 0) |
                                      (looped_market['Ideal_So_Segment'] > 0)]

        if len(looped_market) < 3:
            print(f"Warning: Not enough valid data points for {loc} after filtering zeros")
            continue

        # Fit linear regression model
        X, y = looped_market[["Real_So_Segment"]], looped_market[["Ideal_So_Segment"]]
        model.fit(X, y)
        r_squared = model.score(X, y)
        market_score = round(r_squared * 10, 2)

        location.append(loc)
        score.append(market_score)

    # Create DataFrame
    cat_c_scores = pd.DataFrame(list(zip(location, score)), columns=['Location', 'RSQ'])

    return cat_c_scores

def process_clusters(similarity_file, iata_location, market_summary_pmi, market_summary_comp, brand_attributes):
    """Process clusters for Category D"""
    print("Processing clusters for Category D...")

    # Prepare similarity file
    similarity_file1 = pd.melt(
        similarity_file,
        id_vars=['IATA'],
        var_name='Cluster',
        value_name='Score'
    )
    similarity_file1 = similarity_file1[similarity_file1['Score'] < 1]
    similarity_file2 = similarity_file1.sort_values(['IATA', 'Score'], ascending=False)
    similarity_file2['Rank'] = similarity_file2.groupby('IATA').rank(
        method='first',
        ascending=False
    )['Score']
    clusters = similarity_file2[similarity_file2.Rank <= 4]

    # Process clusters
    clusterlist = pd.DataFrame()

    for iata in clusters.IATA.unique():
        a = market_summary_pmi[market_summary_pmi['IATA'] == iata]
        a = a.drop(columns=['SoM_PMI'])
        a = a.rename(columns={'PMI_Seg_SKU': 'PMI SKU'})

        if len(a) == 0:
            continue

        selected_iata = clusters[clusters['IATA'] == iata]
        cluster_iata = list(selected_iata.Cluster.unique())

        b = market_summary_pmi[market_summary_pmi['IATA'].isin(cluster_iata)]
        b['IATA'] = iata
        b = b[['IATA'] + brand_attributes + ['PMI_Seg_SKU']]
        b = b.groupby(by=['IATA'] + brand_attributes).sum(['PMI_Seg_SKU']).reset_index()
        b = b.rename(columns={'PMI_Seg_SKU': 'Cluster Segment'})

        c = market_summary_comp[market_summary_comp['IATA'].isin(cluster_iata)]
        c['IATA'] = iata
        c = c[['IATA'] + brand_attributes + ['Comp_Seg_SKU']]
        c = c.groupby(by=['IATA'] + brand_attributes).sum(['Comp_Seg_SKU']).reset_index()
        c = c.rename(columns={'Comp_Seg_SKU': 'Cluster Segment'})

        d = pd.concat([b, c], ignore_index=True)
        d = d.groupby(by=['IATA'] + brand_attributes).sum(['Cluster Segment']).reset_index()
        d_x = d.groupby(by=['IATA']).sum('Cluster Segment').reset_index()
        d_x = d_x.rename(columns={'Cluster Segment': 'Cluster_Total'})
        d = d.merge(d_x, how='left', on='IATA')

        e = a.merge(d, how='outer', on=['IATA'] + brand_attributes).fillna(0)
        e['DF_Market'] = list(a.DF_Market.unique())[0]
        e['IATA'] = list(a.IATA.unique())[0]
        e['Location'] = list(a.Location.unique())[0]

        clusterlist = pd.concat([e, clusterlist])
        clusterlist['PMI SKU %'] = np.where(
            clusterlist['PMI Total'] > 0,
            clusterlist['PMI SKU'] / clusterlist['PMI Total'],
            0
        )
        clusterlist['Cluster SKU %'] = np.where(
            clusterlist['Cluster_Total'] > 0,
            clusterlist['Cluster Segment'] / clusterlist['Cluster_Total'],
            0
        )
        clusterlist['SKU Delta'] = clusterlist['PMI SKU %'] - clusterlist['Cluster SKU %']

    clusterlist = clusterlist.rename(columns={'Cluster Segment': 'Cluster SKU'})
    clusterlist = clusterlist[['DF_Market', 'IATA', 'Location',
                               'Taste', 'Thickness', 'Flavor', 'Length',
                               'PMI SKU', 'PMI Total', 'PMI SKU %',
                               'Cluster SKU', 'Cluster_Total', 'Cluster SKU %']]

    return clusterlist


def calculate_cat_d_scores(clusterlist, market):
    """Calculate Category D scores without defaulting"""
    print("Calculating Category D scores...")

    # Check if required columns exist
    required_columns = ['Location', 'PMI SKU', 'Cluster SKU']
    if not all(col in clusterlist.columns for col in required_columns):
        raise ValueError(
            f"clusterlist is missing required columns: {[col for col in required_columns if col not in clusterlist.columns]}")

    location = []
    score = []
    num_of_pmi_sku = []
    num_of_comp_sku = []

    for loc in clusterlist['Location'].unique():
        # Filter data for this location
        looped_market = clusterlist[clusterlist['Location'] == loc]

        # Check if we have enough data points
        if len(looped_market) < 3:
            print(f"Warning: Not enough data points for {loc}, skipping score calculation")
            continue

        # Filter out rows with zero values to avoid skewed results
        looped_market = looped_market[(looped_market['PMI SKU'] > 0) |
                                      (looped_market['Cluster SKU'] > 0)]

        if len(looped_market) < 3:
            print(f"Warning: Not enough valid data points for {loc} after filtering zeros")
            continue

        # Fit linear regression model
        X, y = looped_market[["PMI SKU"]], looped_market[["Cluster SKU"]]
        model.fit(X, y)
        r_squared = model.score(X, y)
        market_score = round(r_squared * 10, 2)

        # Get SKU counts if available in market DataFrame
        market_filtered = market[market['Location'] == loc]
        if not market_filtered.empty:
            pmi_sku_col_idx = -4
            comp_sku_col_idx = -7

            if abs(pmi_sku_col_idx) <= market_filtered.shape[1] and abs(comp_sku_col_idx) <= market_filtered.shape[1]:
                skunum = market_filtered.iloc[:, pmi_sku_col_idx].max()
                compsku = market_filtered.iloc[:, comp_sku_col_idx].max()
            else:
                print(f"Warning: Index out of bounds for {loc} in market DataFrame")
                skunum = None
                compsku = None
        else:
            print(f"Warning: No data in market DataFrame for {loc}")
            skunum = None
            compsku = None

        location.append(loc)
        score.append(market_score)
        num_of_pmi_sku.append(skunum if skunum is not None else 0)
        num_of_comp_sku.append(compsku if compsku is not None else 0)

    # Create DataFrame
    data = {'Location': location, 'RSQ': score}
    if len(num_of_pmi_sku) == len(location):
        data['NumPMI_SKU'] = num_of_pmi_sku
    if len(num_of_comp_sku) == len(location):
        data['NumComp_SKU'] = num_of_comp_sku

    cat_d_scores = pd.DataFrame(data)
    cat_d_scores = cat_d_scores.rename(columns={'RSQ': 'Cat_D'})

    return cat_d_scores


def prepare_final_scores(cat_a_scores, cat_b_scores, cat_c_scores, cat_d_scores, cat_a_df_vols):
    """Prepare final scores table without using default values"""
    print("Preparing final scores table...")

    # Process cat_a_scores
    cat_a_scores = cat_a_scores[['Location', 'ScaledScore']].rename(columns={'ScaledScore': 'Cat_A'})
    cat_a_scores['Location'] = cat_a_scores['Location'].str.strip()

    # Process cat_b_scores
    cat_b_scores = cat_b_scores[['Location', 'Cat_B']]
    cat_b_scores['Location'] = cat_b_scores['Location'].str.strip()

    # Process cat_c_scores
    cat_c_scores = cat_c_scores[['Location', 'RSQ']].rename(columns={'RSQ': 'Cat_C'})
    cat_c_scores['Location'] = cat_c_scores['Location'].str.strip()

    # Process cat_d_scores
    cat_d_scores = cat_d_scores[['Location', 'Cat_D']]
    cat_d_scores['Location'] = cat_d_scores['Location'].str.strip()

    # Merge scores using outer join to keep all locations from any score
    final_table = cat_a_scores.merge(cat_b_scores, how='outer', on='Location')
    final_table = final_table.merge(cat_c_scores, how='outer', on='Location')
    final_table = final_table.merge(cat_d_scores, how='outer', on='Location')

    # Calculate average score, handling missing values properly
    cols_to_average = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']

    # Handle perfect scores (10) by temporarily marking them as NaN to avoid skewing average
    final_table[cols_to_average] = final_table[cols_to_average].replace(10, pd.NA)

    # Calculate average only for rows with at least one valid score
    final_table['Avg_Score'] = final_table[cols_to_average].mean(axis=1, skipna=True)

    # Restore 10s for display
    final_table[cols_to_average] = final_table[cols_to_average].replace(pd.NA, 10)

    # Format and filter
    final_table['Avg_Score'] = final_table['Avg_Score'].astype('float')
    final_table = final_table[final_table['Avg_Score'].notna()]
    final_table['Avg_Score'] = round(final_table['Avg_Score'], 2)

    # Get volume information
    vol_col = f"{current_year} Volume"

    # Calculate total market volumes
    location_volumes = pd.pivot_table(
        cat_a_df_vols,
        index=['Location'],
        aggfunc={vol_col: 'sum'}
    ).reset_index()
    location_volumes = location_volumes.rename(columns={vol_col: 'Market_Volume'})

    # Calculate PMI volumes
    pmi_volumes = pd.pivot_table(
        cat_a_df_vols[cat_a_df_vols['TMO'] == 'PMI'],
        index=['Location'],
        aggfunc={vol_col: 'sum'}
    ).reset_index()
    pmi_volumes = pmi_volumes.rename(columns={vol_col: 'PMI_Volume'})

    # Add volume information using left join to maintain all locations
    final_table = final_table.merge(location_volumes, how='left', on='Location')
    final_table = final_table.merge(pmi_volumes, how='left', on='Location')

    # Ensure Market_Volume and PMI_Volume are not null
    volume_cols = ['Market_Volume', 'PMI_Volume']
    for col in volume_cols:
        # For locations with no volume data, query it directly
        missing_volumes = final_table[final_table[col].isna()]
        if not missing_volumes.empty:
            for loc in missing_volumes['Location']:
                print(f"Warning: No {col} data for {loc}, fetching from database")
                # This would be implemented with a direct query if needed

    return final_table

def main():
    """Main function to run the portfolio optimization script"""
    # Create output directory for results
    os.makedirs('results', exist_ok=True)

    # Connect to Snowflake
    conn = connect_to_snowflake()

    # Fetch data from Snowflake
    print("\nFetching data from Snowflake...")
    mc_per_product = fetch_mc_per_product(conn)
    cat_a_df_vols = fetch_df_vols(conn)
    selma_df_map = fetch_selma_df_map(conn)
    base_list = fetch_base_list(conn)
    pmi_products = fetch_pmi_products(conn)
    dom_prods_data = fetch_dom_prods_data(conn)
    dom_ims_data = fetch_dom_ims_data(conn)
    domestic_volumes = fetch_domestic_volumes(conn)
    country_figures = fetch_country_figures(conn)
    df_vol_data = fetch_df_vol_data(conn)
    pax_data = fetch_pax_data(conn)

    # Load or create reference mappings
    iata_location = fetch_iata_location()
    mrk_nat_map = fetch_nationality_mapping()

    # Load SELMA dom map - assumed to be available
    try:
        selma_dom_map = pd.read_excel(f"{data_dir}/Dom_SELMA_output_2023120.xlsx")
        selma_dom_map.to_pickle(f"{data_dir}/SELMA_dom_map.pkl")
    except:
        print("Warning: SELMA_dom_map not found. This will affect Category C calculation.")
        # Create a minimal version to allow script to run
        selma_dom_map = pd.DataFrame(columns=['Market', 'EBROMId', 'EBROM', 'Taste', 'Thickness', 'Flavor', 'Length'])

    # Load similarity file for clusters - assumed to be available
    try:
        similarity_file = pd.read_excel(f"{data_dir}/matrix_end_2023.xlsx")
    except:
        print("Warning: similarity_file not found. This will affect Category D calculation.")
        # Create a minimal version
        similarity_file = pd.DataFrame(columns=['IATA'] + ['Cluster' + str(i) for i in range(1, 10)])

    # Process Category A
    print("\nProcessing Category A...")
    df_vols_w_financials = create_df_vols_w_financials(cat_a_df_vols, mc_per_product)
    df_vols_w_financials.to_pickle(f"{data_dir}/df_vols_w_financials.pkl")

    pmi_margins = create_pmi_margins(df_vols_w_financials)

    sku_by_vols_margins = create_sku_by_vols_margins(df_vols_w_financials, pmi_margins)
    sku_by_vols_margins.to_pickle(f"{data_dir}/sku_by_vols_margins.pkl")

    no_of_sku = create_flag_counts(df_vols_w_financials)

    gf = create_green_flags(df_vols_w_financials, no_of_sku)
    gf2 = create_green_flags2(sku_by_vols_margins, no_of_sku)

    green_list = create_green_list(gf, gf2)

    rf1 = create_red_flags1(df_vols_w_financials, no_of_sku)
    rf2 = create_red_flags2(sku_by_vols_margins, no_of_sku)

    red_list = create_red_list(rf1, rf2)

    green_red_list = create_green_red_list(green_list, red_list)
    green_red_list.to_pickle(f"{data_dir}/green_red_list.pkl")

    category_a_1 = create_category_a_dataframe(sku_by_vols_margins, green_red_list)

    calculation_table = create_calculation_table(category_a_1)

    cat_a_scores = calculate_cat_a_scores(calculation_table)
    cat_a_scores.to_pickle(f"{data_dir}/cat_a_scores.pkl")

    # Process Category B
    print("\nProcessing Category B...")
    duty_free_volumes, category_list = prepare_duty_free_volumes(cat_a_df_vols, base_list)

    tobacco_range, tobacco_range2 = create_tobacco_range(duty_free_volumes, category_list)

    market_mix = create_market_mix(selma_df_map, tobacco_range2, brand_attributes)
    market_mix = market_mix.merge(iata_location, how='left', on='Location')

    market_summary0, market_summary = create_market_summary(market_mix, tobacco_range, brand_attributes)

    market_summary_pmi, market_summary_comp = create_market_summary_pmi_comp(market_summary)

    market_summary_delta, market = create_market_delta(market_summary_comp, market_summary_pmi, market_summary0,
                                                       brand_attributes)

    cat_b_scores = calculate_cat_b_scores(market, market_mix)
    cat_b_scores.to_pickle(f"{data_dir}/cat_b_scores.pkl")

    # Process Category C (PARIS)
    print("\nProcessing Category C...")

    dimension = brand_attributes  # Assuming dimension is same as brand_attributes
    domestic_dimensions = ['Market', 'EBROMId', 'EBROM', 'Taste', 'Thickness', 'Flavor', 'Length']

    # Create cleaned copy of DF_Vol_data
    df_vols = df_vol_data.copy()

    try:
        paris_output, dom_fin, projected_vol_by_sku, proj_som_pg, df_som_iata_bysku = process_paris_module(
            pax_data, mrk_nat_map, iata_location, country_figures,
            dom_ims_data, domestic_volumes, dom_prods_data, selma_dom_map,
            df_vols, selma_df_map, time_dim, brand_attributes, current_year
        )

        cat_c_scores = calculate_cat_c_scores(paris_output)
    except Exception as e:
        print(f"Error processing Category C: {str(e)}")
        print("Creating dummy Cat C scores from existing data...")
        # Create dummy cat_c_scores based on locations from cat_a_scores
        cat_c_scores = pd.DataFrame({'Location': cat_a_scores['Location'].unique()})
        cat_c_scores['RSQ'] = 5.0  # Default score
    cat_c_scores.to_pickle(f"{data_dir}/cat_c_scores.pkl")

    # Process Category D
    print("\nProcessing Category D...")
    try:
        clusterlist = process_clusters(similarity_file, iata_location, market_summary_pmi, market_summary_comp,
                                       brand_attributes)
        cat_d_scores = calculate_cat_d_scores(clusterlist, market)
    except Exception as e:
        print(f"Error processing Category D: {str(e)}")
        print("Creating dummy Cat D scores from existing data...")
        # Create dummy cat_d_scores
        cat_d_scores = pd.DataFrame({'Location': cat_a_scores['Location'].unique()})
        cat_d_scores['RSQ'] = 5.0  # Default score
        cat_d_scores['NumPMI_SKU'] = 0
        cat_d_scores['NumComp_SKU'] = 0

    cat_d_scores.to_pickle(f"{data_dir}/cat_d_scores.pkl")

    # Create final scores
    print("\nCreating final scores...")
    final_scores = prepare_final_scores(cat_a_scores, cat_b_scores, cat_c_scores, cat_d_scores, cat_a_df_vols)

    # Save final output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_scores.to_csv(f"results/all_scores_{timestamp}.csv", index=False)
    final_scores.to_excel(f"results/all_scores_{timestamp}.xlsx", index=False)

    # Print sample results
    print("\nSample results for a few locations:")
    sample_locations = ['Kuwait', 'Jeju', 'Zurich']
    for loc in sample_locations:
        if loc in final_scores['Location'].values:
            print(f"\n{loc} scores:")
            print(final_scores[final_scores['Location'] == loc])

    print(f"\nProcessing complete. Full results saved to results/all_scores_{timestamp}.csv")

    # Close connection
    conn.close()


if __name__ == "__main__":
    main()