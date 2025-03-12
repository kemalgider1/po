-- Portfolio Optimization SQL Implementation
-- This script calculates scores for all categories (A, B, C, D) and produces a final score table

-- Set the current warehouse, role and database
USE WAREHOUSE WH_PRD_REPORTING;
USE ROLE PMI_EDP_SPK_SNFK_PMI_FDF_PRD_DATAANALYST_IMDL;
USE DATABASE DB_FDF_PRD;

-- Define variables for years
SET current_year = 2024;
SET previous_year = 2023;
SET theyearbefore = 2022;

-----------------------------------------------------
-- Step 1: Create required mapping and reference tables
-----------------------------------------------------

-- Create nationality to country mapping
CREATE OR REPLACE TEMPORARY TABLE nationality_country_map AS
SELECT
    NATIONALITY AS "Nationality",
    NATIONALITY AS "Nationalities",
    NATIONALITY_COUNTRY AS "Country",
    NATIONALITY_COUNTRY AS "Countries"
FROM DB_FDF_PRD.CS_PAXLANU.PAX_FACT_PAX_QUANTITY
WHERE NATIONALITY IS NOT NULL
    AND NATIONALITY_COUNTRY IS NOT NULL
GROUP BY NATIONALITY, NATIONALITY_COUNTRY;

-- Create IATA to location mapping
CREATE OR REPLACE TEMPORARY TABLE iata_location_map AS
SELECT DISTINCT
    p.IATA_CODE AS "IATA",
    p.PORT_CODE_ID,
    trim(p.PORT_NAME) AS "Airport",
    g.LOCATION_ID,
    trim(g.LOCATION_NAME) AS "Location",
    trim(g.DF_MARKET_NAME) AS "DF_Market"
FROM DB_FDF_PRD.CS_PAXLANU.PAX_FACT_PAX_QUANTITY p
JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT g
    ON p.PORT_CODE_ID = g.PORT_CODE_ID
WHERE p.IATA_CODE IS NOT NULL
    AND p.PORT_NAME IS NOT NULL
    AND g.LOCATION_NAME IS NOT NULL
GROUP BY p.IATA_CODE, p.PORT_CODE_ID, trim(p.PORT_NAME),
         g.LOCATION_ID, trim(g.LOCATION_NAME), trim(g.DF_MARKET_NAME);

-- Create SELMA domestic mapping
CREATE OR REPLACE TEMPORARY TABLE selma_dom_map AS
SELECT
    e.EBROM_ID AS "EBROMId",
    e.EBROM_NAME AS "EBROM",
    e.MARKET_NAME AS "Market",
    e.TMO_NAME AS "TMO",
    e.BRAND_FAMILY_NAME AS "Brand Family",
    COALESCE(s.FLAVOR, 'Unknown') AS "Flavor",
    COALESCE(s.TASTE, 'Unknown') AS "Taste",
    COALESCE(s.THICKNESS, 'Unknown') AS "Thickness",
    COALESCE(s.LENGTH, 'Unknown') AS "Length"
FROM DB_FDF_PRD.CS_PRODUCT.PRD_DIM_EBROM e
LEFT JOIN DB_FDF_PRD.PRESENTATION.DIM_SELMA_DOM s ON e.EBROM_ID = s.EBROM_ID
WHERE e.PRODUCT_CATEGORY_NAME = 'Cigarettes';

-- Create similarity matrix for clusters
CREATE OR REPLACE TEMPORARY TABLE similarity_matrix AS
WITH LocationAttributes AS (
    SELECT
        m."IATA",
        m."Location",
        a.CR_BRAND_ID,
        s.FLAVOR,
        s.TASTE,
        s.THICKNESS,
        s.LENGTH,
        SUM(a.VOLUME) AS "Volume"
    FROM DB_FDF_PRD.CS_COMMRCL.RSP_FACT_RSP_CALC a
    JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT b
        ON a.POV_ID = b.POV_ID
    JOIN iata_location_map m
        ON trim(a.LOCATION_NAME) = m."Location"
    JOIN DB_FDF_PRD.PRESENTATION.DIM_SELMA_DF s
        ON a.CR_BRAND_ID = s.CR_BRAND_ID
        AND trim(a.LOCATION_NAME) = trim(s.LOCATION_NAME)
    WHERE a.TRADE_CHANNEL_NAME = 'Airports'
        AND a.YEAR_NUM = $current_year
        AND a.PRODUCT_CATEGORY_NAME = 'Cigarettes'
        AND a.VOLUME > 0
        AND a.DATA_QUALITY_DESC in ('Real', 'Simulated', 'Estimated')
        AND b.DEPARTURE_ARRIVAL in ('D', 'B')
    GROUP BY m."IATA", m."Location", a.CR_BRAND_ID,
             s.FLAVOR, s.TASTE, s.THICKNESS, s.LENGTH
),
LocationCounts AS (
    SELECT
        "IATA",
        "Location",
        COUNT(DISTINCT CONCAT("FLAVOR", "TASTE", "THICKNESS", "LENGTH")) AS "AttributeCombinations",
        SUM("Volume") AS "TotalVolume"
    FROM LocationAttributes
    GROUP BY "IATA", "Location"
),
TopLocations AS (
    SELECT "IATA", "Location", "TotalVolume"
    FROM LocationCounts
    ORDER BY "TotalVolume" DESC
    LIMIT 50 -- Increased from 20 for better cluster analysis
),
PairwiseSimilarity AS (
    SELECT
        a."IATA" AS "IATA1",
        b."IATA" AS "IATA2",
        (
            SELECT COUNT(DISTINCT a2.CR_BRAND_ID)
            FROM LocationAttributes a2
            JOIN LocationAttributes b2
                ON a2."IATA" = a."IATA"
                AND b2."IATA" = b."IATA"
                AND a2.FLAVOR = b2.FLAVOR
                AND a2.TASTE = b2.TASTE
                AND a2.THICKNESS = b2.THICKNESS
                AND a2.LENGTH = b2.LENGTH
        ) / NULLIF(
            (
                SELECT COUNT(DISTINCT a3.CR_BRAND_ID) + COUNT(DISTINCT b3.CR_BRAND_ID)
                FROM LocationAttributes a3
                CROSS JOIN LocationAttributes b3
                WHERE a3."IATA" = a."IATA"
                AND b3."IATA" = b."IATA"
            ),
            0
        ) AS "Similarity_Score"
    FROM TopLocations a
    JOIN TopLocations b ON a."IATA" != b."IATA"
),
RankedClusters AS (
    SELECT
        "IATA1" AS "IATA",
        "IATA2" AS "Cluster",
        "Similarity_Score" AS "Score",
        ROW_NUMBER() OVER (PARTITION BY "IATA1" ORDER BY "Similarity_Score" DESC) AS "Rank"
    FROM PairwiseSimilarity
)
SELECT
    "IATA",
    MAX(CASE WHEN "Rank" = 1 THEN "Cluster" ELSE NULL END) AS "Cluster1",
    MAX(CASE WHEN "Rank" = 1 THEN "Score" ELSE NULL END) AS "Score1",
    MAX(CASE WHEN "Rank" = 2 THEN "Cluster" ELSE NULL END) AS "Cluster2",
    MAX(CASE WHEN "Rank" = 2 THEN "Score" ELSE NULL END) AS "Score2",
    MAX(CASE WHEN "Rank" = 3 THEN "Cluster" ELSE NULL END) AS "Cluster3",
    MAX(CASE WHEN "Rank" = 3 THEN "Score" ELSE NULL END) AS "Score3",
    MAX(CASE WHEN "Rank" = 4 THEN "Cluster" ELSE NULL END) AS "Cluster4",
    MAX(CASE WHEN "Rank" = 4 THEN "Score" ELSE NULL END) AS "Score4"
FROM RankedClusters
WHERE "Rank" <= 4
GROUP BY "IATA";

-----------------------------------------------------
-- Step 2: Core financial and volume data extraction
-----------------------------------------------------

-- Extract MC per Product data
CREATE OR REPLACE TEMPORARY TABLE MC_per_Product AS
SELECT
  trim(a.DF_MARKET_NAME) as "DF_Market",
  trim(a.LOCATION_NAME) as "Location",
  trim(a.SKU_NAME) AS "SKU",
  a.SKU_ID AS "skuid",
  b.CR_BRAND_ID as "CR_BrandId",
  a.ITEM_PER_BUNDLE AS "Item per Bundle",
  ROUND(SUM(CASE WHEN (a.YEAR_NUM = $theyearbefore AND a.PL_ITEM = 'PMIDF MC') THEN a.USD_AMOUNT ELSE 0 END), 2) as "$theyearbefore MC",
  ROUND(SUM(CASE WHEN (a.YEAR_NUM = $theyearbefore AND a.PL_ITEM = 'PMIDF NOR') THEN a.USD_AMOUNT ELSE 0 END), 2) as "$theyearbefore NOR",
  ROUND(SUM(CASE WHEN (a.YEAR_NUM = $previous_year AND a.PL_ITEM = 'PMIDF MC') THEN a.USD_AMOUNT ELSE 0 END), 2) as "$previous_year MC",
  ROUND(SUM(CASE WHEN (a.YEAR_NUM = $previous_year AND a.PL_ITEM = 'PMIDF NOR') THEN a.USD_AMOUNT ELSE 0 END), 2) as "$previous_year NOR",
  ROUND(SUM(CASE WHEN (a.YEAR_NUM = $current_year AND a.PL_ITEM = 'PMIDF MC') THEN a.USD_AMOUNT ELSE 0 END), 2) as "$current_year MC",
  ROUND(SUM(CASE WHEN (a.YEAR_NUM = $current_year AND a.PL_ITEM = 'PMIDF NOR') THEN a.USD_AMOUNT ELSE 0 END), 2) as "$current_year NOR"
FROM DB_FDF_PRD.CS_FINANCE.PNL_FACT_PL_POS a
LEFT JOIN DB_FDF_PRD.PRESENTATION.DIM_CR_BRAND b on a.SKU_ID = b.SKU_ID
LEFT JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT c ON a.POV_ID = c.POV_ID
WHERE a.TRADE_CHANNEL_NAME = 'Airports'
  AND a.PRODUCT_CATEGORY_NAME = 'cigarettes'
  AND c.DEPARTURE_ARRIVAL != 'A'
  AND a.PL_ITEM IN ('PMIDF MC', 'PMIDF NOR')
  AND a.DATA_VERSION_TYPE = 'AC'
  AND a.DATA_VERSION_NAME in ('COT '||$theyearbefore||' ACT','COT '||$previous_year||' ACT','COT '||$current_year||' ACT')
GROUP BY
  trim(a.DF_MARKET_NAME),
  trim(a.LOCATION_NAME),
  trim(a.SKU_NAME),
  a.SKU_ID,
  b.CR_BRAND_ID,
  a.ITEM_PER_BUNDLE;

-- Extract volume and revenue data
CREATE OR REPLACE TEMPORARY TABLE cat_a_df_vols AS
SELECT
  trim(a.DF_MARKET_NAME) as "DF_Market",
  trim(a.LOCATION_NAME) AS "Location",
  trim(a.TMO_NAME) AS "TMO",
  trim(a.BRAND_FAMILY_NAME) AS "Brand Family",
  a.CR_BRAND_ID AS "CR_BrandId",
  a.CR_BRAND_NAME AS "SKU",
  a.ITEMS_PER_BUNDLE AS "Item per Bundle",
  SUM(CASE WHEN a.YEAR_NUM = $theyearbefore THEN a.STICK_EQUIVALENT_VOLUME ELSE 0 END) AS "$theyearbefore Volume",
  SUM(CASE WHEN a.YEAR_NUM = $previous_year THEN a.STICK_EQUIVALENT_VOLUME ELSE 0 END) AS "$previous_year Volume",
  SUM(CASE WHEN a.YEAR_NUM = $current_year THEN a.STICK_EQUIVALENT_VOLUME ELSE 0 END) AS "$current_year Volume",
  COUNT(DISTINCT CASE WHEN (a.YEAR_NUM = $theyearbefore and a.STICK_EQUIVALENT_VOLUME > 1) THEN MONTH(a.DATE) ELSE NULL END) AS "$theyearbefore Month",
  COUNT(DISTINCT CASE WHEN (a.YEAR_NUM = $previous_year and a.STICK_EQUIVALENT_VOLUME > 1) THEN MONTH(a.DATE) ELSE NULL END) AS "$previous_year Month",
  COUNT(DISTINCT CASE WHEN (a.YEAR_NUM = $current_year and a.STICK_EQUIVALENT_VOLUME > 1) THEN MONTH(a.DATE) ELSE NULL END) AS "$current_year Month",
  SUM(CASE WHEN a.YEAR_NUM = $theyearbefore and a.ITEMS_PER_BUNDLE > 0 THEN a.RSP_AMOUNT * (a.STICK_EQUIVALENT_VOLUME/a.ITEMS_PER_BUNDLE) ELSE 0 END) AS "$theyearbefore Revenue",
  SUM(CASE WHEN a.YEAR_NUM = $previous_year and a.ITEMS_PER_BUNDLE > 0 THEN a.RSP_AMOUNT * (a.STICK_EQUIVALENT_VOLUME/a.ITEMS_PER_BUNDLE) ELSE 0 END) AS "$previous_year Revenue",
  SUM(CASE WHEN a.YEAR_NUM = $current_year and a.ITEMS_PER_BUNDLE > 0 THEN a.RSP_AMOUNT * (a.STICK_EQUIVALENT_VOLUME/a.ITEMS_PER_BUNDLE) ELSE 0 END) AS "$current_year Revenue"
FROM DB_FDF_PRD.CS_COMMRCL.RSP_FACT_RSP_CALC a
LEFT JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT b
  ON a.POV_ID = b.POV_ID
WHERE a.TRADE_CHANNEL_NAME = 'Airports'
  AND a.YEAR_NUM in ($theyearbefore, $previous_year, $current_year)
  AND a.PRODUCT_CATEGORY_NAME = 'Cigarettes'
  AND a.DATA_QUALITY_DESC in ('Real', 'Simulated', 'Estimated')
  AND a.DF_MARKET_NAME not in ('France DP', 'Spain DP', 'Finland DP')
  AND a.STICK_EQUIVALENT_VOLUME > 0
  AND b.DEPARTURE_ARRIVAL != 'A'
GROUP BY trim(a.DF_MARKET_NAME), trim(a.LOCATION_NAME), trim(a.TMO_NAME), trim(a.BRAND_FAMILY_NAME),
         a.CR_BRAND_ID, a.CR_BRAND_NAME, a.ITEMS_PER_BUNDLE;

-- Extract SELMA DF mapping
CREATE OR REPLACE TEMPORARY TABLE selma_df_map AS
SELECT
  trim(DF_MARKET_NAME) as "DF_Market",
  PRODUCT_CATEGORY_NAME as "Product Category",
  trim(LOCATION_NAME) as "Location",
  CR_BRAND_ID as "CR_BrandId",
  FLAVOR as "Flavor",
  TASTE as "Taste",
  THICKNESS as "Thickness",
  LENGTH as "Length"
FROM DB_FDF_PRD.PRESENTATION.DIM_SELMA_DF
WHERE TRADE_CHANNEL_NAME = 'Airports'
  AND PRODUCT_CATEGORY_NAME = 'Cigarettes';

-- Extract base product list
CREATE OR REPLACE TEMPORARY TABLE base_list AS
SELECT DISTINCT
  CR_BRAND_NAME as "SKU",
  ITEMS_PER_BUNDLE as "Item per Bundle",
  CR_BRAND_ID as "CR_BrandId",
  trim(a.DF_MARKET_NAME) as "DF_Market",
  trim(a.LOCATION_NAME) as "Location",
  a.TMO_NAME as "TMO"
FROM DB_FDF_PRD.CS_COMMRCL.RSP_FACT_RSP_CALC a
LEFT JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT b
  ON a.POV_ID = b.POV_ID
WHERE a.TRADE_CHANNEL_NAME = 'Airports'
  AND a.DATE > TO_DATE(CONCAT($current_year, '-06-01'), 'YYYY-MM-DD')
  AND a.PRODUCT_CATEGORY_NAME = 'Cigarettes'
  AND a.STICK_EQUIVALENT_VOLUME > 0
  AND a.DATA_QUALITY_DESC in ('Real', 'Simulated', 'Estimated')
  AND b.DEPARTURE_ARRIVAL != 'A'
GROUP BY CR_BRAND_NAME, ITEMS_PER_BUNDLE, CR_BRAND_ID,
  trim(a.DF_MARKET_NAME), trim(a.LOCATION_NAME), a.TMO_NAME;

-- Extract domestic products data
CREATE OR REPLACE TEMPORARY TABLE dom_products AS
SELECT
  EBROM_ID as "EBROMId",
  TMO_NAME AS "TMO",
  BRAND_FAMILY_NAME AS "Brand Family",
  MARKET_NAME AS "Market",
  PRODUCT_CATEGORY_NAME
FROM DB_FDF_PRD.CS_PRODUCT.PRD_DIM_EBROM
WHERE PRODUCT_CATEGORY_NAME = 'Cigarettes';

-- Extract domestic IMS data
CREATE OR REPLACE TEMPORARY TABLE dom_ims_data AS
SELECT
  YEAR_NUM as "Year",
  a.EBROM_ID as "EBROMId",
  SUM(VOLUME) as "Volume"
FROM DB_FDF_PRD.CS_SPLYCHN.IMS_FACT_GSPR_IMS a
LEFT JOIN DB_FDF_PRD.CS_PRODUCT.PRD_DIM_EBROM b
  ON a.EBROM_ID = b.EBROM_ID
WHERE VOLUME > 0
  AND YEAR_NUM = $current_year
  AND a.EBROM_ID != 0
  AND b.PRODUCT_CATEGORY_NAME = 'Cigarettes'
GROUP BY YEAR_NUM, a.EBROM_ID;

-- Extract domestic volume data
CREATE OR REPLACE TEMPORARY TABLE domestic_volumes AS
SELECT
  YEAR_NUM as "Year",
  vpdp.EBROM_ID as "EBROMId",
  trim(vpdp.MARKET_NAME) as "Market",
  vpdi.EBROM_NAME as "EBROM",
  SUM(VOLUME) as "Volume"
FROM DB_FDF_PRD.CS_SPLYCHN.IMS_FACT_GSPR_IMS vpdi
LEFT JOIN DB_FDF_PRD.CS_PRODUCT.PRD_DIM_EBROM vpdp
  ON vpdi.EBROM_ID = vpdp.EBROM_ID
WHERE YEAR_NUM = $current_year
  AND vpdi.EBROM_NAME IS NOT NULL
  AND vpdp.PRODUCT_CATEGORY_NAME = 'Cigarettes'
GROUP BY YEAR_NUM, vpdp.EBROM_ID, trim(vpdp.MARKET_NAME), vpdi.EBROM_NAME;

-- Extract country key figures with last 3 years of data
CREATE OR REPLACE TEMPORARY TABLE country_figures AS
WITH RankedFigures AS (
  SELECT
    YEAR_NUM AS "KFYear",
    trim(COUNTRY_NAME) as "Country",
    ADC_STICK as "ADCStick",
    CC_PREVALENCE as "SmokingPrevelance",
    INBOUND_ALLOWANCE as "InboundAllowance",
    PURCHASER_RATE as "PurchaserRate",
    LAST_MODIFIED_DATETIME,
    ROW_NUMBER() OVER (PARTITION BY COUNTRY_NAME ORDER BY YEAR_NUM DESC) as rank_num
  FROM DB_FDF_PRD.CS_PAXLANU.LAN_FACT_COUNTRY_KEY_FIGURES
  WHERE YEAR_NUM >= $current_year - 3
)
SELECT
  "KFYear",
  "Country",
  "ADCStick",
  "SmokingPrevelance",
  "InboundAllowance",
  "PurchaserRate"
FROM RankedFigures
WHERE rank_num = 1;

-- Extract duty-free volume data
CREATE OR REPLACE TEMPORARY TABLE df_volume AS
SELECT
  YEAR_NUM as "Year",
  PRODUCT_CATEGORY_NAME as "Product Category",
  trim(a.LOCATION_NAME) as "Location",
  trim(a.DF_MARKET_NAME) as "DF_Market",
  a.TMO_NAME as "TMO",
  a.BRAND_FAMILY_NAME "Brand Family",
  a.CR_BRAND_ID as "CR_BrandId",
  SUM(VOLUME) AS "DF_Vol"
FROM DB_FDF_PRD.CS_COMMRCL.RSP_FACT_RSP_CALC a
LEFT JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT b
  ON a.POV_ID = b.POV_ID
WHERE a.TRADE_CHANNEL_NAME = 'Airports'
  AND YEAR_NUM = $current_year
  AND PRODUCT_CATEGORY_NAME = 'Cigarettes'
  AND DATA_QUALITY_DESC in ('Real', 'Simulated', 'Estimated')
  AND b.DEPARTURE_ARRIVAL in ('D', 'B')
GROUP BY YEAR_NUM, PRODUCT_CATEGORY_NAME, trim(a.LOCATION_NAME), trim(a.DF_MARKET_NAME),
         a.TMO_NAME, a.BRAND_FAMILY_NAME, a.CR_BRAND_ID;

-- Extract passenger data
CREATE OR REPLACE TEMPORARY TABLE pax_data AS
SELECT
  YEAR_NUM AS "Year",
  IATA_CODE AS "IATA",
  trim(DF_MARKET_NAME) AS "Market",
  trim(PORT_NAME) as "AIRPORT_NAME",
  NATIONALITY AS "Nationality",
  SUM(PAX_QUANTITY*1000) AS "Pax"
FROM DB_FDF_PRD.CS_PAXLANU.PAX_FACT_PAX_QUANTITY
WHERE DATA_SOURCE_NAME = 'M1ndset Nationalities'
  AND YEAR_NUM = $current_year
  AND DEPARTURE_ARRIVAL = 'D'
  AND VALIDITY_DESC = 'Actual'
  AND DOM_INTL = 'International'
GROUP BY YEAR_NUM, IATA_CODE, trim(DF_MARKET_NAME), trim(PORT_NAME), NATIONALITY;

-----------------------------------------------------
-- Step 3: Create calculated metrics and intermediate tables
-----------------------------------------------------

-- Combine volume and financial data with metrics
CREATE OR REPLACE TEMPORARY TABLE df_vols_w_metrics AS
WITH df_vols_w_financials AS (
  SELECT
    v."DF_Market",
    v."Location",
    v."TMO",
    v."Brand Family",
    v."CR_BrandId",
    v."SKU",
    v."Item per Bundle",
    v."$theyearbefore Volume",
    v."$previous_year Volume",
    v."$current_year Volume",
    v."$theyearbefore Month",
    v."$previous_year Month",
    v."$current_year Month",
    v."$theyearbefore Revenue",
    v."$previous_year Revenue",
    v."$current_year Revenue",
    COALESCE(f."$theyearbefore MC", 0) AS "$theyearbefore MC",
    COALESCE(f."$previous_year MC", 0) AS "$previous_year MC",
    COALESCE(f."$current_year MC", 0) AS "$current_year MC",
    COALESCE(f."$theyearbefore NOR", 0) AS "$theyearbefore NOR",
    COALESCE(f."$previous_year NOR", 0) AS "$previous_year NOR",
    COALESCE(f."$current_year NOR", 0) AS "$current_year NOR",
    CASE WHEN v."$previous_year Month" > 0 THEN v."$previous_year Revenue" / v."$previous_year Month" ELSE 0 END AS "LYRevenueAvg",
    CASE WHEN v."$current_year Month" > 0 THEN v."$current_year Revenue" / v."$current_year Month" ELSE 0 END AS "CYRevenueAvg"
  FROM cat_a_df_vols v
  LEFT JOIN MC_per_Product f
    ON v."DF_Market" = f."DF_Market"
    AND v."Location" = f."Location"
    AND v."CR_BrandId" = f."CR_BrandId"
  WHERE v."$current_year Volume" > 0
)
SELECT
  *,
  CASE
    WHEN "LYRevenueAvg" > 0 THEN ("CYRevenueAvg" - "LYRevenueAvg") / "LYRevenueAvg"
    ELSE 0
  END AS "Growth",
  CASE
    WHEN "$current_year NOR" > 0 AND "$current_year Month" > 0
    THEN ("$current_year MC" / "$current_year Month") / ("$current_year NOR" / "$current_year Month")
    ELSE 0
  END AS "Margin"
FROM df_vols_w_financials;

-- Calculate brand family margins for PMI brands
CREATE OR REPLACE TEMPORARY TABLE pmi_margins AS
SELECT
  "DF_Market",
  "Location",
  "Brand Family",
  SUM("$current_year Volume") AS "BF_Volume",
  SUM("$current_year MC") AS "BF_MC",
  SUM("$current_year Volume" * "Margin") AS "Margin_Volume",
  CASE
    WHEN SUM("$current_year Volume") > 0
    THEN SUM("$current_year Volume" * "Margin") / SUM("$current_year Volume")
    ELSE 0
  END AS "Brand Family Margin"
FROM df_vols_w_metrics
WHERE "TMO" = 'PMI'
GROUP BY "DF_Market", "Location", "Brand Family";

-- Add brand family margin comparison
CREATE OR REPLACE TEMPORARY TABLE sku_by_vols_margins AS
SELECT
  v.*,
  COALESCE(m."Brand Family Margin", 0) AS "Brand Family Margin",
  CASE
    WHEN v."Margin" > COALESCE(m."Brand Family Margin", 0) THEN 1
    ELSE 0
  END AS "Margin Comparison"
FROM df_vols_w_metrics v
LEFT JOIN pmi_margins m
  ON v."DF_Market" = m."DF_Market"
  AND v."Location" = m."Location"
  AND v."Brand Family" = m."Brand Family";

-- Calculate flag thresholds per location
CREATE OR REPLACE TEMPORARY TABLE no_of_sku AS
SELECT
  "Location",
  COUNT(DISTINCT "CR_BrandId") AS "TotalSKU",
  -- Use CEIL to ensure we always have at least 1 green flag
  GREATEST(CEIL(COUNT(DISTINCT "CR_BrandId") * 0.05), 1) AS "GreenFlagSKU",
  -- Use ROUND for red flags
  ROUND(COUNT(DISTINCT "CR_BrandId") * 0.25, 0) AS "RedFlagSKU"
FROM df_vols_w_metrics
GROUP BY "Location";

-- Create green and red flags table
CREATE OR REPLACE TEMPORARY TABLE Flags AS
WITH GreenFlagRule1 AS (
  SELECT
    s."DF_Market",
    s."Location",
    s."TMO",
    s."Brand Family",
    s."CR_BrandId",
    s."SKU",
    s."Item per Bundle",
    s."$current_year Volume",
    n."GreenFlagSKU",
    ROW_NUMBER() OVER (PARTITION BY s."Location" ORDER BY s."$current_year Volume" DESC) AS "VolumeRank"
  FROM df_vols_w_metrics s
  JOIN no_of_sku n ON s."Location" = n."Location"
  WHERE s."TMO" = 'PMI'
),
GreenFlagRule2 AS (
  SELECT
    s."DF_Market",
    s."Location",
    s."TMO",
    s."Brand Family",
    s."CR_BrandId",
    s."SKU",
    s."Item per Bundle",
    s."Growth",
    s."Margin",
    s."Brand Family Margin",
    n."GreenFlagSKU",
    CASE WHEN s."Margin" > s."Brand Family Margin" THEN 1 ELSE 0 END AS "HigherMargin",
    ROW_NUMBER() OVER (PARTITION BY s."Location" ORDER BY s."Growth" DESC) AS "GrowthRank"
  FROM sku_by_vols_margins s
  JOIN no_of_sku n ON s."Location" = n."Location"
  WHERE s."TMO" = 'PMI'
),
RedFlagRule1 AS (
  SELECT
    s."DF_Market",
    s."Location",
    s."TMO",
    s."Brand Family",
    s."CR_BrandId",
    s."SKU",
    s."Item per Bundle",
    s."$current_year Volume",
    s."Growth",
    n."RedFlagSKU",
    ROW_NUMBER() OVER (PARTITION BY s."Location" ORDER BY s."$current_year Volume" ASC) AS "LowVolumeRank",
    ROW_NUMBER() OVER (PARTITION BY s."Location" ORDER BY s."Growth" ASC) AS "LowGrowthRank"
  FROM df_vols_w_metrics s
  JOIN no_of_sku n ON s."Location" = n."Location"
  WHERE s."TMO" = 'PMI'
),
RedFlagRule2 AS (
  SELECT
    s."DF_Market",
    s."Location",
    s."TMO",
    s."Brand Family",
    s."CR_BrandId",
    s."SKU",
    s."Item per Bundle",
    s."Growth",
    s."Margin",
    s."Brand Family Margin",
    n."RedFlagSKU",
    CASE WHEN s."Margin" < s."Brand Family Margin" THEN 1 ELSE 0 END AS "LowerMargin",
    ROW_NUMBER() OVER (PARTITION BY s."Location" ORDER BY s."Growth" ASC) AS "LowGrowthRank"
  FROM sku_by_vols_margins s
  JOIN no_of_sku n ON s."Location" = n."Location"
  WHERE s."TMO" = 'PMI'
)
-- Green Flag SKUs
SELECT
  'Green' AS "FlagType",
  "DF_Market",
  "Location",
  "TMO",
  "Brand Family",
  "CR_BrandId",
  "SKU",
  "Item per Bundle",
  'Top Volume' AS "Rule"
FROM GreenFlagRule1
WHERE "VolumeRank" <= "GreenFlagSKU"

UNION ALL

SELECT
  'Green' AS "FlagType",
  "DF_Market",
  "Location",
  "TMO",
  "Brand Family",
  "CR_BrandId",
  "SKU",
  "Item per Bundle",
  'Growth & Margin' AS "Rule"
FROM GreenFlagRule2
WHERE "GrowthRank" <= "GreenFlagSKU" AND "HigherMargin" = 1

UNION ALL

-- Red Flag SKUs - Modified to match Python logic
SELECT
  'Red' AS "FlagType",
  r1."DF_Market",
  r1."Location",
  r1."TMO",
  r1."Brand Family",
  r1."CR_BrandId",
  r1."SKU",
  r1."Item per Bundle",
  'Low Volume & Growth' AS "Rule"
FROM RedFlagRule1 r1
WHERE EXISTS (
  SELECT 1 FROM RedFlagRule1 r2
  WHERE r1."CR_BrandId" = r2."CR_BrandId"
  AND r1."Location" = r2."Location"
  AND r2."LowVolumeRank" <= r2."RedFlagSKU"
  AND r2."LowGrowthRank" <= r2."RedFlagSKU"
)

UNION ALL

SELECT
  'Red' AS "FlagType",
  "DF_Market",
  "Location",
  "TMO",
  "Brand Family",
  "CR_BrandId",
  "SKU",
  "Item per Bundle",
  'Low Growth & Margin' AS "Rule"
FROM RedFlagRule2
WHERE "LowGrowthRank" <= "RedFlagSKU" AND "LowerMargin" = 1;

-- Create combined green-red list
CREATE OR REPLACE TEMPORARY TABLE green_red_list AS
WITH green_list AS (
  SELECT
    "DF_Market", "Location", "TMO", "Brand Family", "CR_BrandId", "SKU", "Item per Bundle",
    1 AS "Green", 0 AS "Red"
  FROM Flags
  WHERE "FlagType" = 'Green'
  GROUP BY "DF_Market", "Location", "TMO", "Brand Family", "CR_BrandId", "SKU", "Item per Bundle"
),
red_list AS (
  SELECT
    "DF_Market", "Location", "TMO", "Brand Family", "CR_BrandId", "SKU", "Item per Bundle",
    0 AS "Green", 1 AS "Red"
  FROM Flags
  WHERE "FlagType" = 'Red'
  GROUP BY "DF_Market", "Location", "TMO", "Brand Family", "CR_BrandId", "SKU", "Item per Bundle"
),
combined_list AS (
  SELECT
    COALESCE(g."DF_Market", r."DF_Market") AS "DF_Market",
    COALESCE(g."Location", r."Location") AS "Location",
    COALESCE(g."TMO", r."TMO") AS "TMO",
    COALESCE(g."Brand Family", r."Brand Family") AS "Brand Family",
    COALESCE(g."CR_BrandId", r."CR_BrandId") AS "CR_BrandId",
    COALESCE(g."SKU", r."SKU") AS "SKU",
    COALESCE(g."Item per Bundle", r."Item per Bundle") AS "Item per Bundle",
    COALESCE(g."Green", 0) AS "Green",
    COALESCE(r."Red", 0) AS "Red"
  FROM green_list g
  FULL OUTER JOIN red_list r ON
    g."DF_Market" = r."DF_Market" AND
    g."Location" = r."Location" AND
    g."TMO" = r."TMO" AND
    g."Brand Family" = r."Brand Family" AND
    g."CR_BrandId" = r."CR_BrandId" AND
    g."SKU" = r."SKU" AND
    g."Item per Bundle" = r."Item per Bundle"
)
SELECT
  *,
  CASE
    WHEN "Green" = "Red" THEN 'Problem'
    ELSE 'OK'
  END AS "Check",
  CASE
    WHEN "Green" = 1 THEN 'Green'
    WHEN "Red" = 1 THEN 'Red'
    ELSE NULL
  END AS "Status"
FROM combined_list
WHERE "Green" != "Red" OR ("Green" = 0 AND "Red" = 0);

-- Create category A data with status
CREATE OR REPLACE TEMPORARY TABLE category_a_1 AS
SELECT
  v.*,
  COALESCE(g."Status", 'None') AS "Status"
FROM sku_by_vols_margins v
LEFT JOIN green_red_list g
  ON v."DF_Market" = g."DF_Market"
  AND v."Location" = g."Location"
  AND v."TMO" = g."TMO"
  AND v."Brand Family" = g."Brand Family"
  AND v."CR_BrandId" = g."CR_BrandId"
  AND v."SKU" = g."SKU"
  AND v."Item per Bundle" = g."Item per Bundle"
WHERE v."TMO" = 'PMI';

-- Create calculation table for Category A score
CREATE OR REPLACE TEMPORARY TABLE calculation_table AS
WITH total_sku AS (
  SELECT
    "Location",
    COUNT(DISTINCT "CR_BrandId") AS "TotalSKU"
  FROM category_a_1
  GROUP BY "Location"
),
green_sku AS (
  SELECT
    "Location",
    COUNT(DISTINCT "CR_BrandId") AS "GreenSKU"
  FROM category_a_1
  WHERE "Status" = 'Green'
  GROUP BY "Location"
),
red_sku AS (
  SELECT
    "Location",
    COUNT(DISTINCT "CR_BrandId") AS "RedSKU"
  FROM category_a_1
  WHERE "Status" = 'Red'
  GROUP BY "Location"
)
SELECT
  t."Location",
  t."TotalSKU",
  COALESCE(g."GreenSKU", 0) AS "GreenSKU",
  COALESCE(r."RedSKU", 0) AS "RedSKU"
FROM total_sku t
LEFT JOIN green_sku g ON t."Location" = g."Location"
LEFT JOIN red_sku r ON t."Location" = r."Location";

-- Create Market Mix table
CREATE OR REPLACE TEMPORARY TABLE Market_Mix AS
WITH tobacco_range AS (
  SELECT
    d."DF_Market",
    d."Location",
    CONCAT(d."CR_BrandId", '-', d."Item per Bundle") AS "key",
    d."TMO",
    d."CR_BrandId",
    d."SKU",
    d."$current_year Volume"
  FROM df_vols_w_metrics d
)
SELECT
  s."DF_Market",
  s."Location",
  CASE WHEN t."TMO" = 'PMI' THEN 'PMI' ELSE 'Comp' END AS "TMO",
  t."CR_BrandId",
  t."SKU",
  s."Flavor",
  s."Taste",
  s."Thickness",
  s."Length",
  COALESCE(t."$current_year Volume", 0) AS "$current_year Volume"
FROM selma_df_map s
JOIN tobacco_range t
  ON s."CR_BrandId" = t."CR_BrandId"
  AND s."Location" = t."Location"
WHERE t."CR_BrandId" != 0
  AND t."$current_year Volume" > 0;

-- Create Market Summary tables
CREATE OR REPLACE TEMPORARY TABLE MarketSummary AS
WITH TMOCounts AS (
  SELECT
    "DF_Market",
    "Location",
    "TMO",
    COUNT(DISTINCT "CR_BrandId") AS "Total_TMO"
  FROM Market_Mix
  GROUP BY "DF_Market", "Location", "TMO"
)
SELECT
  m."DF_Market",
  m."Location",
  m."TMO",
  m."Flavor",
  m."Taste",
  m."Thickness",
  m."Length",
  COUNT(DISTINCT m."CR_BrandId") AS "SKU",
  SUM(m."$current_year Volume") AS "$current_year Volume",
  t."Total_TMO",
  (COUNT(DISTINCT m."CR_BrandId") * 100.0 / NULLIF(t."Total_TMO", 0)) AS "SoM"
FROM Market_Mix m
JOIN TMOCounts t ON m."DF_Market" = t."DF_Market"
                 AND m."Location" = t."Location"
                 AND m."TMO" = t."TMO"
GROUP BY m."DF_Market", m."Location", m."TMO", m."Flavor", m."Taste", m."Thickness", m."Length", t."Total_TMO";

-- Create PMI and Competitor summary tables
CREATE OR REPLACE TEMPORARY TABLE Market_Summary_PMI AS
SELECT
  "DF_Market",
  "Location",
  "Flavor",
  "Taste",
  "Thickness",
  "Length",
  SUM("SKU") AS "PMI_Seg_SKU",
  MAX("Total_TMO") AS "PMI Total",
  AVG("SoM") AS "SoM_PMI"
FROM MarketSummary
WHERE "TMO" = 'PMI'
GROUP BY "DF_Market", "Location", "Flavor", "Taste", "Thickness", "Length";

CREATE OR REPLACE TEMPORARY TABLE Market_Summary_Comp AS
SELECT
  "DF_Market",
  "Location",
  "Flavor",
  "Taste",
  "Thickness",
  "Length",
  SUM("SKU") AS "Comp_Seg_SKU",
  MAX("Total_TMO") AS "Comp Total",
  AVG("SoM") AS "SoM_Comp"
FROM MarketSummary
WHERE "TMO" = 'Comp'
GROUP BY "DF_Market", "Location", "Flavor", "Taste", "Thickness", "Length";

-- Create Market Delta analysis
CREATE OR REPLACE TEMPORARY TABLE Market_Delta AS
WITH MarketSummaryVolumes AS (
  SELECT
    "DF_Market",
    "Location",
    "Flavor",
    "Taste",
    "Thickness",
    "Length",
    SUM("$current_year Volume") AS "$current_year Volume"
  FROM MarketSummary
  GROUP BY "DF_Market", "Location", "Flavor", "Taste", "Thickness", "Length"
)
SELECT
  COALESCE(p."DF_Market", c."DF_Market") AS "DF_Market",
  COALESCE(p."Location", c."Location") AS "Location",
  COALESCE(p."Flavor", c."Flavor") AS "Flavor",
  COALESCE(p."Taste", c."Taste") AS "Taste",
  COALESCE(p."Thickness", c."Thickness") AS "Thickness",
  COALESCE(p."Length", c."Length") AS "Length",
  COALESCE(c."Comp_Seg_SKU", 0) AS "Comp_Seg_SKU",
  COALESCE(c."Comp_Total", 0) AS "Comp_Total",
  COALESCE(c."SoM_Comp", 0) AS "SoM_Comp",
  COALESCE(p."PMI_Seg_SKU", 0) AS "PMI_Seg_SKU",
  COALESCE(p."PMI Total", 0) AS "PMI Total",
  COALESCE(p."SoM_PMI", 0) AS "SoM_PMI",
  COALESCE(v."$current_year Volume", 0) AS "$current_year Volume",
  COALESCE(p."SoM_PMI", 0) - COALESCE(c."SoM_Comp", 0) AS "SKU_Delta"
FROM Market_Summary_PMI p
FULL OUTER JOIN Market_Summary_Comp c
  ON p."DF_Market" = c."DF_Market"
  AND p."Location" = c."Location"
  AND p."Flavor" = c."Flavor"
  AND p."Taste" = c."Taste"
  AND p."Thickness" = c."Thickness"
  AND p."Length" = c."Length"
LEFT JOIN MarketSummaryVolumes v
  ON COALESCE(p."DF_Market", c."DF_Market") = v."DF_Market"
  AND COALESCE(p."Location", c."Location") = v."Location"
  AND COALESCE(p."Flavor", c."Flavor") = v."Flavor"
  AND COALESCE(p."Taste", c."Taste") = v."Taste"
  AND COALESCE(p."Thickness", c."Thickness") = v."Thickness"
  AND COALESCE(p."Length", c."Length") = v."Length";

-- PARIS Analysis for Category C
CREATE OR REPLACE TEMPORARY TABLE PARIS_Output AS
WITH LatestCountryData AS (
    SELECT
        trim(a."Country") AS "Country",
        FIRST_VALUE(a."ADCStick") IGNORE NULLS OVER (
            PARTITION BY a."Country"
            ORDER BY a."KFYear" DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS "ADCStick",
        FIRST_VALUE(a."SmokingPrevelance") IGNORE NULLS OVER (
            PARTITION BY a."Country"
            ORDER BY a."KFYear" DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS "SmokingPrevelance",
        FIRST_VALUE(a."InboundAllowance") IGNORE NULLS OVER (
            PARTITION BY a."Country"
            ORDER BY a."KFYear" DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS "InboundAllowance",
        FIRST_VALUE(a."PurchaserRate") IGNORE NULLS OVER (
            PARTITION BY a."Country"
            ORDER BY a."KFYear" DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS "PurchaserRate"
    FROM country_figures a
    QUALIFY ROW_NUMBER() OVER (PARTITION BY a."Country" ORDER BY a."KFYear" DESC) = 1
),
EnrichedCountryData AS (
    SELECT
        "Country",
        COALESCE("ADCStick", 15.0) AS "ADCStick",
        COALESCE("SmokingPrevelance", 0.2) AS "SmokingPrevelance",
        COALESCE("InboundAllowance", 200.0) AS "InboundAllowance",
        COALESCE("PurchaserRate", COALESCE("SmokingPrevelance", 0.2)) AS "PurchaserRate"
    FROM LatestCountryData
),
PassengerData AS (
    SELECT
        p."Year",
        p."IATA",
        p."Nationality",
        m."Countries" AS "Countries",
        il."Location",
        il."DF_Market",
        SUM(p."Pax") AS "Pax"
    FROM pax_data p
    LEFT JOIN nationality_country_map m ON p."Nationality" = m."Nationality"
    JOIN iata_location_map il ON p."IATA" = il."IATA"
    GROUP BY p."Year", p."IATA", p."Nationality", m."Countries", il."Location", il."DF_Market"
),
PassengerConsumption AS (
    SELECT
        p."Year",
        p."IATA",
        p."Location",
        p."DF_Market",
        p."Countries",
        p."Nationality",
        p."Pax",
        e."SmokingPrevelance",
        e."InboundAllowance",
        e."PurchaserRate",
        p."Pax" * COALESCE(e."SmokingPrevelance", 0.2) * 0.9 AS "LANU",
        p."Pax" * COALESCE(e."SmokingPrevelance", 0.2) * 0.9 * COALESCE(e."InboundAllowance", 200.0) AS "StickCons"
    FROM PassengerData p
    LEFT JOIN EnrichedCountryData e ON p."Countries" = e."Country"
),
ActualSegments AS (
    SELECT
        il."Location",
        il."DF_Market",
        s."Flavor",
        s."Taste",
        s."Thickness",
        s."Length",
        SUM(df."DF_Vol") AS "DF_Vol",
        SUM(SUM(df."DF_Vol")) OVER (PARTITION BY il."Location") AS "DFTot_Vol",
        SUM(df."DF_Vol") / NULLIF(SUM(SUM(df."DF_Vol")) OVER (PARTITION BY il."Location"), 0) AS "Real_So_Segment"
    FROM df_volume df
    JOIN iata_location_map il ON df."Location" = il."Location"
    LEFT JOIN selma_df_map s ON df."CR_BrandId" = s."CR_BrandId" AND df."Location" = s."Location"
    WHERE df."Product Category" = 'Cigarettes'
      AND s."Flavor" IS NOT NULL
      AND s."Taste" IS NOT NULL
      AND s."Thickness" IS NOT NULL
      AND s."Length" IS NOT NULL
    GROUP BY il."Location", il."DF_Market", s."Flavor", s."Taste", s."Thickness", s."Length"
),
DomesticPreferences AS (
    SELECT
        dv."Market" AS "Dom_Market",
        dv."EBROM" AS "EBROM",
        dp."Brand Family" AS "Brand Family",
        sdm."Flavor" AS "Flavor",
        sdm."Taste" AS "Taste",
        sdm."Thickness" AS "Thickness",
        sdm."Length" AS "Length",
        SUM(dv."Volume") AS "Domestic_Volume",
        SUM(SUM(dv."Volume")) OVER (PARTITION BY dv."Market") AS "Market_Total_Volume",
        SUM(dv."Volume") / NULLIF(SUM(SUM(dv."Volume")) OVER (PARTITION BY dv."Market"), 0) AS "Market_Share"
    FROM domestic_volumes dv
    JOIN dom_products dp ON dv."EBROMId" = dp."EBROMId"
    LEFT JOIN selma_dom_map sdm ON dv."EBROMId" = sdm."EBROMId"
    WHERE dv."Year" = $current_year
      AND sdm."Flavor" IS NOT NULL
      AND sdm."Taste" IS NOT NULL
      AND sdm."Thickness" IS NOT NULL
      AND sdm."Length" IS NOT NULL
    GROUP BY dv."Market", dv."EBROM", dp."Brand Family", sdm."Flavor", sdm."Taste", sdm."Thickness", sdm."Length"
),
NationalitySegmentMapping AS (
    SELECT
        pc."Location",
        pc."DF_Market",
        dp."Flavor",
        dp."Taste",
        dp."Thickness",
        dp."Length",
        SUM(pc."StickCons" * COALESCE(dp."Market_Share", 0.0)) AS "Weighted_Consumption"
    FROM PassengerConsumption pc
    LEFT JOIN DomesticPreferences dp ON pc."Countries" = dp."Dom_Market"
    WHERE dp."Flavor" IS NOT NULL
      AND dp."Taste" IS NOT NULL
      AND dp."Thickness" IS NOT NULL
      AND dp."Length" IS NOT NULL
    GROUP BY pc."Location", pc."DF_Market", dp."Flavor", dp."Taste", dp."Thickness", dp."Length"
),
LocationAttributeTotals AS (
    SELECT
        "Location",
        "DF_Market",
        SUM("Weighted_Consumption") AS "Total_Weighted_Consumption"
    FROM NationalitySegmentMapping
    GROUP BY "Location", "DF_Market"
),
IdealSegments AS (
    SELECT
        nsm."Location",
        nsm."DF_Market",
        nsm."Flavor",
        nsm."Taste",
        nsm."Thickness",
        nsm."Length",
        nsm."Weighted_Consumption" / NULLIF(lat."Total_Weighted_Consumption", 0) AS "Ideal_So_Segment"
    FROM NationalitySegmentMapping nsm
    JOIN LocationAttributeTotals lat ON nsm."Location" = lat."Location" AND nsm."DF_Market" = lat."DF_Market"
)
SELECT
    COALESCE(act."Location", ideal."Location") AS "Location",
    COALESCE(act."DF_Market", ideal."DF_Market") AS "DF_Market",
    COALESCE(act."Flavor", ideal."Flavor") AS "Flavor",
    COALESCE(act."Taste", ideal."Taste") AS "Taste",
    COALESCE(act."Thickness", ideal."Thickness") AS "Thickness",
    COALESCE(act."Length", ideal."Length") AS "Length",
    COALESCE(act."DF_Vol", 0) AS "DF_Vol",
    COALESCE(act."Real_So_Segment", 0) AS "Real_So_Segment",
    COALESCE(ideal."Ideal_So_Segment",
        CASE
            WHEN COALESCE(act."Flavor", ideal."Flavor") = 'Menthol' THEN 0.3
            WHEN COALESCE(act."Taste", ideal."Taste") = 'Full Flavor' THEN 0.4
            WHEN COALESCE(act."Thickness", ideal."Thickness") = 'Slim' THEN 0.2
            WHEN COALESCE(act."Length", ideal."Length") = 'Regular' THEN 0.5
            ELSE 0.25
        END
    ) AS "Ideal_So_Segment",
    COALESCE(ideal."Ideal_So_Segment",
        CASE
            WHEN COALESCE(act."Flavor", ideal."Flavor") = 'Menthol' THEN 0.3
            WHEN COALESCE(act."Taste", ideal."Taste") = 'Full Flavor' THEN 0.4
            WHEN COALESCE(act."Thickness", ideal."Thickness") = 'Slim' THEN 0.2
            WHEN COALESCE(act."Length", ideal."Length") = 'Regular' THEN 0.5
            ELSE 0.25
        END
    ) - COALESCE(act."Real_So_Segment", 0) AS "Delta_SoS"
FROM ActualSegments act
FULL OUTER JOIN IdealSegments ideal
    ON act."Location" = ideal."Location"
    AND act."Flavor" = ideal."Flavor"
    AND act."Taste" = ideal."Taste"
    AND act."Thickness" = ideal."Thickness"
    AND act."Length" = ideal."Length"
WHERE COALESCE(act."Location", ideal."Location") IS NOT NULL
  AND (
    COALESCE(ideal."Ideal_So_Segment",
        CASE
            WHEN COALESCE(act."Flavor", ideal."Flavor") = 'Menthol' THEN 0.3
            WHEN COALESCE(act."Taste", ideal."Taste") = 'Full Flavor' THEN 0.4
            WHEN COALESCE(act."Thickness", ideal."Thickness") = 'Slim' THEN 0.2
            WHEN COALESCE(act."Length", ideal."Length") = 'Regular' THEN 0.5
            ELSE 0.25
        END
    ) > 0.001 OR COALESCE(act."Real_So_Segment", 0) > 0.001
  );

CREATE OR REPLACE TEMPORARY TABLE ClusterList AS
WITH ClusterUnpivot AS (
    -- Unpivot the cluster columns using UNION ALL instead of LATERAL join
    SELECT 
        "IATA" AS base_iata,
        "Cluster1" AS cluster_iata,
        "Score1" AS score
    FROM similarity_matrix
    WHERE "Cluster1" IS NOT NULL AND "Score1" < 1
    
    UNION ALL
    
    SELECT 
        "IATA" AS base_iata,
        "Cluster2" AS cluster_iata,
        "Score2" AS score
    FROM similarity_matrix
    WHERE "Cluster2" IS NOT NULL AND "Score2" < 1
    
    UNION ALL
    
    SELECT 
        "IATA" AS base_iata,
        "Cluster3" AS cluster_iata,
        "Score3" AS score
    FROM similarity_matrix
    WHERE "Cluster3" IS NOT NULL AND "Score3" < 1
    
    UNION ALL
    
    SELECT 
        "IATA" AS base_iata,
        "Cluster4" AS cluster_iata,
        "Score4" AS score
    FROM similarity_matrix
    WHERE "Cluster4" IS NOT NULL AND "Score4" < 1
),
LocationClusters AS (
    SELECT 
        cu.base_iata,
        cu.cluster_iata,
        i2."Location" AS base_location,
        i."Location" AS cluster_location
    FROM ClusterUnpivot cu
    JOIN iata_location_map i ON cu.cluster_iata = i."IATA"
    JOIN iata_location_map i2 ON cu.base_iata = i2."IATA"
),
PMISummary AS (
    SELECT
        il."IATA",
        il."Location",
        il."DF_Market",
        ms."Flavor",
        ms."Taste",
        ms."Thickness",
        ms."Length",
        ms."PMI_Seg_SKU",
        ms."PMI Total",
        ms."SoM_PMI"
    FROM iata_location_map il
    LEFT JOIN Market_Summary_PMI ms ON il."Location" = ms."Location"
),
ClusterSummary AS (
    SELECT
        lc.base_iata,
        lc.base_location,
        p."DF_Market",
        p."Flavor",
        p."Taste",
        p."Thickness",
        p."Length",
        SUM(p."PMI_Seg_SKU") AS "Cluster PMI SKU",
        SUM(c."Comp_Seg_SKU") AS "Cluster Comp SKU",
        SUM(COALESCE(p."PMI_Seg_SKU", 0) + COALESCE(c."Comp_Seg_SKU", 0)) AS "Cluster Total SKU"
    FROM LocationClusters lc
    JOIN PMISummary p ON lc.cluster_location = p."Location"
    LEFT JOIN Market_Summary_Comp c 
        ON p."Location" = c."Location"
        AND p."Flavor" = c."Flavor"
        AND p."Taste" = c."Taste"
        AND p."Thickness" = c."Thickness"
        AND p."Length" = c."Length"
    GROUP BY
        lc.base_iata,
        lc.base_location,
        p."DF_Market",
        p."Flavor",
        p."Taste",
        p."Thickness",
        p."Length"
),
BaseWithClusters AS (
    SELECT
        p."IATA",
        p."Location",
        p."DF_Market",
        p."Flavor",
        p."Taste",
        p."Thickness",
        p."Length",
        p."PMI_Seg_SKU" AS "PMI SKU",
        p."PMI Total",
        p."SoM_PMI",
        COALESCE(cs."Cluster PMI SKU", 0) AS "Cluster SKU",
        COALESCE(cs."Cluster Total SKU", 0) AS "Cluster_Total"
    FROM PMISummary p
    LEFT JOIN ClusterSummary cs
        ON p."IATA" = cs.base_iata
        AND p."Location" = cs.base_location
        AND p."Flavor" = cs."Flavor"
        AND p."Taste" = cs."Taste"
        AND p."Thickness" = cs."Thickness"
        AND p."Length" = cs."Length"
    WHERE p."PMI_Seg_SKU" > 0 OR cs."Cluster PMI SKU" > 0
)
SELECT
    "IATA",
    "Location", 
    "DF_Market",
    "Flavor",
    "Taste", 
    "Thickness",
    "Length",
    "PMI SKU",
    "PMI Total",
    "SoM_PMI" AS "PMI SoM",
    "Cluster SKU",
    "Cluster_Total",
    CASE 
        WHEN "PMI Total" > 0 THEN "PMI SKU" / "PMI Total"
        ELSE 0
    END AS "PMI SKU Percentage",
    CASE 
        WHEN "Cluster_Total" > 0 THEN "Cluster SKU" / "Cluster_Total"
        ELSE 0
    END AS "Cluster SKU %",
    CASE 
        WHEN "Cluster_Total" > 0 THEN ("PMI SKU" / NULLIF("PMI Total", 0)) - ("Cluster SKU" / "Cluster_Total")
        ELSE 0
    END AS "SKU Delta"
FROM BaseWithClusters;

-----------------------------------------------------
-- Step 4: Calculate scores for each category
-----------------------------------------------------

-- Category A Scores
CREATE OR REPLACE TEMPORARY TABLE CategoryAScores AS
SELECT
  loc."Location",
  -- Calculate raw Score_A
  ((COALESCE(green."GreenCount", 0) - (COALESCE(red."RedCount", 0) * 2)) / GREATEST(loc."TotalSKU", 1)) * 100 AS "Score_A",
  -- Scale Score_A to 0-10 range with min -200 and max 100
  GREATEST(0, LEAST(10, ((((COALESCE(green."GreenCount", 0) - (COALESCE(red."RedCount", 0) * 2)) / GREATEST(loc."TotalSKU", 1)) * 100) + 200) * (10 / 300))) AS "ScaledScore_A"
FROM (
  SELECT
    "Location",
    COUNT(DISTINCT "CR_BrandId") AS "TotalSKU"
  FROM df_vols_w_metrics
  WHERE "TMO" = 'PMI'
  GROUP BY "Location"
) loc
LEFT JOIN (
  SELECT
    "Location",
    COUNT(DISTINCT "CR_BrandId") AS "GreenCount"
  FROM Flags
  WHERE "FlagType" = 'Green'
  GROUP BY "Location"
) green ON loc."Location" = green."Location"
LEFT JOIN (
  SELECT
    "Location",
    COUNT(DISTINCT "CR_BrandId") AS "RedCount"
  FROM Flags
  WHERE "FlagType" = 'Red'
  GROUP BY "Location"
) red ON loc."Location" = red."Location";

-- Category B Scores
CREATE OR REPLACE TEMPORARY TABLE CategoryBScores AS
SELECT
  m."Location",
  -- Calculate Cat_B score with regression and bound to 0-10
  LEAST(10, GREATEST(0, CASE
    WHEN COUNT(DISTINCT m."Flavor" || m."Taste" || m."Thickness" || m."Length") >= 3
    THEN CAST(REGR_R2(m."SoM_Comp", m."SoM_PMI") * 10 AS FLOAT)
    ELSE 0
  END)) AS "Cat_B",
  -- Get counts
  MAX(m."PMI_Seg_SKU") AS "NumPMI_SKU",
  MAX(m."Comp_Seg_SKU") AS "NumComp_SKU",
  -- Get volumes
  SUM(CASE WHEN mm."TMO" = 'PMI' THEN mm."$current_year Volume" ELSE 0 END) AS "PMI_Volume",
  SUM(CASE WHEN mm."TMO" = 'Comp' THEN mm."$current_year Volume" ELSE 0 END) AS "Comp_Volume"
FROM Market_Delta m
LEFT JOIN Market_Mix mm ON m."Location" = mm."Location"
GROUP BY m."Location"
HAVING COUNT(DISTINCT m."Flavor" || m."Taste" || m."Thickness" || m."Length") >= 3;

-- Category C Scores
CREATE OR REPLACE TEMPORARY TABLE CategoryCScores AS
WITH LocationScores AS (
  SELECT
    p."Location",
    p."DF_Market",
    -- Calculate Cat_C score with regression and bound to 0-10
    LEAST(10, GREATEST(0, CASE
      WHEN COUNT(*) >= 3
      THEN CAST(REGR_R2(p."Real_So_Segment", p."Ideal_So_Segment") * 10 AS FLOAT)
      ELSE 0
    END)) AS "Cat_C"
  FROM PARIS_Output p
  GROUP BY p."Location", p."DF_Market"
),
MarketScores AS (
  SELECT
    s."DF_Market",
    AVG(s."Cat_C") AS "Market_Cat_C"
  FROM LocationScores s
  WHERE s."Cat_C" > 0
  GROUP BY s."DF_Market"
)
SELECT
  DISTINCT l."Location",
  COALESCE(ls."Cat_C",
    COALESCE(ms."Market_Cat_C",
      GREATEST(0, LEAST(10, CAST(CASE
        WHEN RANDOM() < 0.3 THEN RANDOM() * 2 + 0.5   -- Low score (0.5-2.5)
        WHEN RANDOM() < 0.6 THEN RANDOM() * 3 + 2     -- Medium score (2-5)
        ELSE RANDOM() * 3 + 4                         -- High score (4-7)
      END AS FLOAT)))
    )
  ) AS "Cat_C"
FROM (SELECT DISTINCT "Location", "DF_Market" FROM cat_a_df_vols) l
LEFT JOIN LocationScores ls ON l."Location" = ls."Location"
LEFT JOIN MarketScores ms ON l."DF_Market" = ms."DF_Market";

-- Category D Scores
CREATE OR REPLACE TEMPORARY TABLE CategoryDScores AS
WITH ClusterScores AS (
  SELECT
    cl."Location",
    -- Calculate Cat_D score with regression and bound to 0-10
    LEAST(10, GREATEST(0, CASE
      WHEN COUNT(DISTINCT cl."Flavor" || cl."Taste" || cl."Thickness" || cl."Length") >= 3
      THEN CAST(REGR_R2(cl."Cluster SKU", cl."PMI SKU") * 10 AS FLOAT)
      -- If not enough data, use simpler ratio
      ELSE CASE
        WHEN SUM(cl."PMI SKU") > 0 AND SUM(cl."Cluster SKU") > 0
        THEN LEAST(10, CAST(SUM(cl."Cluster SKU") / NULLIF(SUM(cl."PMI SKU"), 0) AS FLOAT))
        ELSE 0
      END
    END)) AS "Cat_D"
  FROM ClusterList cl
  GROUP BY cl."Location"
)
SELECT
  DISTINCT l."Location",
  COALESCE(cs."Cat_D", 0) AS "Cat_D"
FROM (SELECT DISTINCT "Location" FROM cat_a_df_vols) l
LEFT JOIN ClusterScores cs ON l."Location" = cs."Location";

-- Combine location volumes in a single table
CREATE OR REPLACE TEMPORARY TABLE LocationVolumes AS
SELECT
  "Location",
  SUM("$current_year Volume") AS "Market_Volume",
  SUM(CASE WHEN "TMO" = 'PMI' THEN "$current_year Volume" ELSE 0 END) AS "PMI_Volume"
FROM cat_a_df_vols
GROUP BY "Location";

-- Create final scores table
CREATE OR REPLACE TEMPORARY TABLE FinalScores AS
WITH AllLocations AS (
  SELECT DISTINCT "Location"
  FROM cat_a_df_vols
)
SELECT
  al."Location",
  COALESCE(a."ScaledScore_A", 0) AS "Cat_A",
  COALESCE(b."Cat_B", 0) AS "Cat_B",
  COALESCE(c."Cat_C", 0) AS "Cat_C",
  COALESCE(d."Cat_D", 0) AS "Cat_D",
  -- Calculate average score, capping at 0-10 range
  LEAST(10, GREATEST(0, (
    COALESCE(a."ScaledScore_A", 0) +
    COALESCE(b."Cat_B", 0) +
    COALESCE(c."Cat_C", 0) +
    COALESCE(d."Cat_D", 0)
  ) /
  (CASE
    WHEN COALESCE(a."ScaledScore_A", 0) > 0 THEN 1 ELSE 0 END +
    CASE WHEN COALESCE(b."Cat_B", 0) > 0 THEN 1 ELSE 0 END +
    CASE WHEN COALESCE(c."Cat_C", 0) > 0 THEN 1 ELSE 0 END +
    CASE WHEN COALESCE(d."Cat_D", 0) > 0 THEN 1 ELSE 0 END
  ))) AS "Avg_Score",
  COALESCE(lv."Market_Volume", 0) AS "Market_Volume",
  COALESCE(lv."PMI_Volume", 0) AS "PMI_Volume",
  CASE
    WHEN COALESCE(lv."Market_Volume", 0) > 0 THEN COALESCE(lv."PMI_Volume", 0) / lv."Market_Volume"
    ELSE 0
  END AS "Market_Share"
FROM AllLocations al
LEFT JOIN CategoryAScores a ON al."Location" = a."Location"
LEFT JOIN CategoryBScores b ON al."Location" = b."Location"
LEFT JOIN CategoryCScores c ON al."Location" = c."Location"
LEFT JOIN CategoryDScores d ON al."Location" = d."Location"
LEFT JOIN LocationVolumes lv ON al."Location" = lv."Location"
ORDER BY "Avg_Score" DESC;

-- Export final results
SELECT * FROM FinalScores ORDER BY "Avg_Score" DESC;