-- Set the current warehouse, role and database
USE WAREHOUSE WH_PRD_REPORTING;
USE ROLE PMI_EDP_SPK_SNFK_PMI_FDF_PRD_DATAANALYST_IMDL;
USE DATABASE DB_FDF_PRD;

-- Define variables for years
SET current_year = 2024;
SET previous_year = 2023;
SET theyearbefore = 2022;

-- Step 1: Create required mapping tables
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


select * from nationality_country_map

---------------------

-- CORRECT IATA to Location Mapping
-- Instead of a direct mapping, we need to establish the proper relationship
-- between IATA codes (airports) and locations (cities)

CREATE OR REPLACE TEMPORARY TABLE iata_location_map AS
SELECT DISTINCT
    p.IATA_CODE AS "IATA",
    p.PORT_CODE_ID,
    trim(p.PORT_NAME) AS "Airport", -- Keep the actual airport name
    g.LOCATION_ID,
    trim(g.LOCATION_NAME) AS "Location", -- This is the city
    trim(g.DF_MARKET_NAME) AS "DF_Market" -- Market associated with the location
FROM DB_FDF_PRD.CS_PAXLANU.PAX_FACT_PAX_QUANTITY p
JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT g
    -- Join on port code which is the connection between airport and location
    ON p.PORT_CODE_ID = g.PORT_CODE_ID
WHERE p.IATA_CODE IS NOT NULL
    AND p.PORT_NAME IS NOT NULL
    AND g.LOCATION_NAME IS NOT NULL
GROUP BY p.IATA_CODE, p.PORT_CODE_ID, trim(p.PORT_NAME),
         g.LOCATION_ID, trim(g.LOCATION_NAME), trim(g.DF_MARKET_NAME);

select * from iata_location_map;


-- Now we have a proper mapping between airport codes and city locations
---------------------

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

select * from selma_dom_map

---------------------

CREATE OR REPLACE TEMPORARY TABLE similarity_matrix AS
WITH LocationAttributes AS (
    SELECT
        m."IATA",
        m."Location", -- City location from mapping table
        a.CR_BRAND_ID,
        s.FLAVOR,
        s.TASTE,
        s.THICKNESS,
        s.LENGTH,
        SUM(a.VOLUME) AS "Volume"
    FROM DB_FDF_PRD.CS_COMMRCL.RSP_FACT_RSP_CALC a
    JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT b
        ON a.POV_ID = b.POV_ID
    -- Use the mapping table instead of direct equality
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
    LIMIT 20
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


select * from similarity_matrix;

---------------------

-- Step 2: Financial metrics
CREATE OR REPLACE TEMPORARY TABLE MC_per_Product AS
SELECT
  trim(a.DF_MARKET_NAME) as "DF_Market",
  trim(a.LOCATION_NAME) as "Location",
  b.CR_BRAND_ID as "CR_BrandId",
  SUM(CASE WHEN a.YEAR_NUM = $theyearbefore AND a.PL_ITEM = 'PMIDF NOR' THEN a.USD_AMOUNT ELSE 0 END) as "$theyearbefore NOR",
  SUM(CASE WHEN a.YEAR_NUM = $previous_year AND a.PL_ITEM = 'PMIDF NOR' THEN a.USD_AMOUNT ELSE 0 END) as "$previous_year NOR",
  SUM(CASE WHEN a.YEAR_NUM = $current_year AND a.PL_ITEM = 'PMIDF NOR' THEN a.USD_AMOUNT ELSE 0 END) as "$current_year NOR",
  SUM(CASE WHEN a.YEAR_NUM = $theyearbefore AND a.PL_ITEM = 'PMIDF MC' THEN a.USD_AMOUNT ELSE 0 END) as "$theyearbefore MC",
  SUM(CASE WHEN a.YEAR_NUM = $previous_year AND a.PL_ITEM = 'PMIDF MC' THEN a.USD_AMOUNT ELSE 0 END) as "$previous_year MC",
  SUM(CASE WHEN a.YEAR_NUM = $current_year AND a.PL_ITEM = 'PMIDF MC' THEN a.USD_AMOUNT ELSE 0 END) as "$current_year MC"
FROM DB_FDF_PRD.CS_FINANCE.PNL_FACT_PL_POS a
LEFT JOIN DB_FDF_PRD.PRESENTATION.DIM_CR_BRAND b ON a.SKU_ID = b.SKU_ID
LEFT JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT c ON a.POV_ID = c.POV_ID
WHERE a.TRADE_CHANNEL_NAME = 'Airports'
  AND a.YEAR_NUM in ($theyearbefore, $previous_year, $current_year)
  AND a.PRODUCT_CATEGORY_NAME = 'Cigarettes'
  AND c.DEPARTURE_ARRIVAL != 'A'
GROUP BY trim(a.DF_MARKET_NAME), trim(a.LOCATION_NAME), b.CR_BRAND_ID;

select * from MC_per_Product

---------------------

-- Step 3: Volume and revenue metrics
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

select * from cat_a_df_vols

---------------------

-- Step 4: SELMA DF mapping
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

select * from selma_df_map

---------------------

-- Step 5: Base product list
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
GROUP BY CR_BRAND_NAME, ITEMS_PER_BUNDLE, CR_BRAND_ID, trim(a.DF_MARKET_NAME), trim(a.LOCATION_NAME), a.TMO_NAME;

select * from base_list

---------------------

-- Step 6: Domestic product data
CREATE OR REPLACE TEMPORARY TABLE dom_products AS
SELECT
  EBROM_ID as "EBROMId",
  TMO_NAME AS "TMO",
  BRAND_FAMILY_NAME AS "Brand Family",
  MARKET_NAME, PRODUCT_CATEGORY_NAME
FROM DB_FDF_PRD.CS_PRODUCT.PRD_DIM_EBROM;


select * from dom_products

---------------------

-- Step 7: Domestic IMS data
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

select * from dom_ims_data
---------------------

-- Step 8: Domestic volume data
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

select * from domestic_volumes

---------------------
-- Step 9: Country key figures
CREATE OR REPLACE TEMPORARY TABLE country_figures AS
SELECT
  YEAR_NUM AS "KFYear",
  trim(COUNTRY_NAME) as "Country",
  ADC_STICK as "ADCStick",
  CC_PREVALENCE as "SmokingPrevelance",
  INBOUND_ALLOWANCE as "InboundAllowance",
  PURCHASER_RATE as "PurchaserRate"
FROM DB_FDF_PRD.CS_PAXLANU.LAN_FACT_COUNTRY_KEY_FIGURES;

select * from country_figures
---------------------

-- Step 10: Duty-free volume data
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


select * from df_volume
---------------------

-- Step 11: Passenger data
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

select * from pax_data
---------------------

-- Step 12: Combine volume and financial data
CREATE OR REPLACE TEMPORARY TABLE df_vols_w_financials AS
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
  COALESCE(f."$theyearbefore NOR", 0) AS "$theyearbefore NOR",
  COALESCE(f."$previous_year NOR", 0) AS "$previous_year NOR",
  COALESCE(f."$current_year NOR", 0) AS "$current_year NOR",
  COALESCE(f."$theyearbefore MC", 0) AS "$theyearbefore MC",
  COALESCE(f."$previous_year MC", 0) AS "$previous_year MC",
  COALESCE(f."$current_year MC", 0) AS "$current_year MC",
  CASE WHEN v."$previous_year Month" > 0 THEN v."$previous_year Revenue" / v."$previous_year Month" ELSE 0 END AS "LYRevenueAvg",
  CASE WHEN v."$current_year Month" > 0 THEN v."$current_year Revenue" / v."$current_year Month" ELSE 0 END AS "CYRevenueAvg"
FROM cat_a_df_vols v
LEFT JOIN MC_per_Product f ON v."DF_Market" = f."DF_Market" AND v."Location" = f."Location" AND v."CR_BrandId" = f."CR_BrandId"
WHERE v."$current_year Volume" > 0;


select * from df_vols_w_financials
---------------------

-- Step 13: Calculate growth and margin
CREATE OR REPLACE TEMPORARY TABLE df_vols_w_metrics AS
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


select * from df_vols_w_metrics
---------------------

-- Step 14: Calculate brand family margins
CREATE OR REPLACE TEMPORARY TABLE pmi_margins AS
SELECT
  "DF_Market",
  "Location",
  "Brand Family",
  SUM("$current_year Volume") AS "BF_Volume",
  SUM("$current_year Volume" * "Margin") AS "Margin_Volume",
  CASE
    WHEN SUM("$current_year Volume") > 0
    THEN SUM("$current_year Volume" * "Margin") / SUM("$current_year Volume")
    ELSE 0
  END AS "Brand Family Margin"
FROM df_vols_w_metrics
WHERE "TMO" = 'PMI'
GROUP BY "DF_Market", "Location", "Brand Family";


select * from pmi_margins
---------------------

-- Step 15: Add brand family margin comparison
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


select * from sku_by_vols_margins
---------------------

-- Step 16: Calculate flag thresholds
CREATE OR REPLACE TEMPORARY TABLE no_of_sku AS
SELECT
  "Location",
  COUNT(*) AS "TotalSKU",
  CEIL(COUNT(*) * 0.05) AS "GreenFlagSKU",
  ROUND(COUNT(*) * 0.25, 0) AS "RedFlagSKU"
FROM df_vols_w_metrics
GROUP BY "Location";


select * from no_of_sku
---------------------

-- Step 17: Create green and red flags
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

-- Red Flag SKUs
SELECT
  'Red' AS "FlagType",
  "DF_Market",
  "Location",
  "TMO",
  "Brand Family",
  "CR_BrandId",
  "SKU",
  "Item per Bundle",
  'Low Volume & Growth' AS "Rule"
FROM RedFlagRule1
WHERE "LowVolumeRank" <= "RedFlagSKU"
  AND "LowGrowthRank" <= "RedFlagSKU"

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

select * from Flags
---------------------

-- Step 18: Create Market Mix table
CREATE OR REPLACE TEMPORARY TABLE Market_Mix AS
WITH ProductAttributes AS (
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
    AND PRODUCT_CATEGORY_NAME = 'Cigarettes'
),
DFVolumes AS (
  SELECT
    YEAR_NUM as "Year",
    PRODUCT_CATEGORY_NAME as "Product Category",
    trim(a.LOCATION_NAME) as "Location",
    trim(a.DF_MARKET_NAME) as "DF_Market",
    a.TMO_NAME as "TMO",
    a.BRAND_FAMILY_NAME "Brand Family",
    a.CR_BRAND_ID as "CR_BrandId",
    SUM(a.VOLUME) AS "DF_Vol"
  FROM DB_FDF_PRD.CS_COMMRCL.RSP_FACT_RSP_CALC a
  LEFT JOIN DB_FDF_PRD.CS_OPERATR.GEO_DIM_TOUCH_POINT b
    ON a.POV_ID = b.POV_ID
  WHERE a.TRADE_CHANNEL_NAME = 'Airports'
    AND a.YEAR_NUM = $current_year
    AND a.PRODUCT_CATEGORY_NAME = 'Cigarettes'
    AND a.DATA_QUALITY_DESC in ('Real', 'Simulated', 'Estimated')
    AND b.DEPARTURE_ARRIVAL in ('D', 'B')
  GROUP BY a.YEAR_NUM, a.PRODUCT_CATEGORY_NAME, trim(a.LOCATION_NAME), trim(a.DF_MARKET_NAME),
           a.TMO_NAME, a.BRAND_FAMILY_NAME, a.CR_BRAND_ID
)
SELECT
  v."DF_Market",
  v."Location",
  v."TMO",
  v."Brand Family",
  v."CR_BrandId",
  p."Flavor",
  p."Taste",
  p."Thickness",
  p."Length",
  v."DF_Vol"
FROM DFVolumes v
JOIN ProductAttributes p ON v."CR_BrandId" = p."CR_BrandId" AND v."Location" = p."Location"
WHERE v."DF_Vol" > 0;


select * from Market_Mix
---------------------
-- Step 19: Create Market Summary tables (Fixed)
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
  SUM(m."DF_Vol") AS "$current_year_Volume",
  t."Total_TMO",
  (COUNT(DISTINCT m."CR_BrandId") * 100.0 / t."Total_TMO") AS "SoM"
FROM Market_Mix m
JOIN TMOCounts t ON m."DF_Market" = t."DF_Market"
                 AND m."Location" = t."Location"
                 AND m."TMO" = t."TMO"
GROUP BY m."DF_Market", m."Location", m."TMO", m."Flavor", m."Taste", m."Thickness", m."Length", t."Total_TMO";


select * from MarketSummary

-- Create PMI and Competitor summaries for segment analysis
CREATE OR REPLACE TEMPORARY TABLE Market_Summary_PMI AS
SELECT
  "DF_Market",
  "Location",
  "Flavor",
  "Taste",
  "Thickness",
  "Length",
  COUNT(DISTINCT "SKU") AS "PMI_Seg_SKU",
  MAX("Total_TMO") AS "PMI Total",
  AVG("SoM") AS "SoM_PMI"
FROM MarketSummary
WHERE "TMO" = 'PMI'
GROUP BY "DF_Market", "Location", "Flavor", "Taste", "Thickness", "Length";


select * from Market_Summary_PMI



CREATE OR REPLACE TEMPORARY TABLE Market_Summary_Comp AS
SELECT
  "DF_Market",
  "Location",
  "Flavor",
  "Taste",
  "Thickness",
  "Length",
  COUNT(DISTINCT "SKU") AS "Comp_Seg_SKU",
  MAX("Total_TMO") AS "Comp_Total",
  AVG("SoM") AS "SoM_Comp"
FROM MarketSummary
WHERE "TMO" != 'PMI'
GROUP BY "DF_Market", "Location", "Flavor", "Taste", "Thickness", "Length";


select * from Market_Summary_Comp
-- Verify the tables




-- Step 20: Create Market Delta analysis (Fixed)

CREATE OR REPLACE TEMPORARY TABLE Market_Delta AS
WITH MarketSummaryVolumes AS (
  SELECT
    "DF_Market",
    "Location",
    "Flavor",
    "Taste",
    "Thickness",
    "Length",
    SUM("$current_year_Volume") AS "$current_year_Volume"
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
  COALESCE(v."$current_year_Volume", 0) AS "$current_year_Volume",
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


select * from Market_Delta
-- Verify the table creation

-- Step 21: PARIS Analysis with proper handling of limited country data

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
        p."Pax" * e."SmokingPrevelance" * 0.9 AS "LANU",
        p."Pax" * e."SmokingPrevelance" * 0.9 * e."InboundAllowance" AS "StickCons"
    FROM PassengerData p
    LEFT JOIN EnrichedCountryData e ON p."Countries" = e."Country"
),
ActualSegments AS (
    SELECT
        trim(df."Location") AS "Location",
        trim(df."DF_Market") AS "DF_Market",
        s."Flavor" AS "Flavor",
        s."Taste" AS "Taste",
        s."Thickness" AS "Thickness",
        s."Length" AS "Length",
        SUM(df."DF_Vol") AS "DF_Vol",
        SUM(SUM(df."DF_Vol")) OVER (PARTITION BY trim(df."Location")) AS "DFTot_Vol",
        SUM(df."DF_Vol") / NULLIF(SUM(SUM(df."DF_Vol")) OVER (PARTITION BY trim(df."Location")), 0) AS "Real_So_Segment"
    FROM df_volume df
    LEFT JOIN selma_df_map s ON df."CR_BrandId" = s."CR_BrandId" AND df."Location" = s."Location"
    WHERE df."Product Category" = 'Cigarettes'
    GROUP BY trim(df."Location"), trim(df."DF_Market"), s."Flavor", s."Taste", s."Thickness", s."Length"
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
    WHERE dp."Flavor" IS NOT NULL AND dp."Taste" IS NOT NULL AND dp."Thickness" IS NOT NULL AND dp."Length" IS NOT NULL
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
),
AllLocations AS (
    SELECT DISTINCT "Location", "DF_Market"
    FROM cat_a_df_vols
),
AllAttributes AS (
    SELECT DISTINCT "Flavor", "Taste", "Thickness", "Length"
    FROM selma_df_map
    WHERE "Flavor" IS NOT NULL AND "Taste" IS NOT NULL AND "Thickness" IS NOT NULL AND "Length" IS NOT NULL
),
CompleteMatrix AS (
    SELECT l."Location", l."DF_Market", a."Flavor", a."Taste", a."Thickness", a."Length"
    FROM AllLocations l
    CROSS JOIN AllAttributes a
)
SELECT
    COALESCE(act."Location", ideal."Location", cm."Location") AS "Location",
    COALESCE(act."DF_Market", ideal."DF_Market", cm."DF_Market") AS "DF_Market",
    COALESCE(act."Flavor", ideal."Flavor", cm."Flavor") AS "Flavor",
    COALESCE(act."Taste", ideal."Taste", cm."Taste") AS "Taste",
    COALESCE(act."Thickness", ideal."Thickness", cm."Thickness") AS "Thickness",
    COALESCE(act."Length", ideal."Length", cm."Length") AS "Length",
    COALESCE(act."DF_Vol", 0) AS "DF_Vol",
    COALESCE(act."Real_So_Segment", 0) AS "Real_So_Segment",
    COALESCE(
        ideal."Ideal_So_Segment",
        CASE
            WHEN COALESCE(act."Flavor", cm."Flavor") = 'Menthol' THEN 0.3
            WHEN COALESCE(act."Taste", cm."Taste") = 'Full Flavor' THEN 0.4
            WHEN COALESCE(act."Thickness", cm."Thickness") = 'Slim' THEN 0.2
            WHEN COALESCE(act."Length", cm."Length") = 'Regular' THEN 0.5
            ELSE 0.25
        END
    ) AS "Ideal_So_Segment",
    COALESCE(
        ideal."Ideal_So_Segment",
        CASE
            WHEN COALESCE(act."Flavor", cm."Flavor") = 'Menthol' THEN 0.3
            WHEN COALESCE(act."Taste", cm."Taste") = 'Full Flavor' THEN 0.4
            WHEN COALESCE(act."Thickness", cm."Thickness") = 'Slim' THEN 0.2
            WHEN COALESCE(act."Length", cm."Length") = 'Regular' THEN 0.5
            ELSE 0.25
        END
    ) - COALESCE(act."Real_So_Segment", 0) AS "Delta_SoS"
FROM CompleteMatrix cm
LEFT JOIN ActualSegments act
    ON cm."Location" = act."Location"
    AND cm."Flavor" = act."Flavor"
    AND cm."Taste" = act."Taste"
    AND cm."Thickness" = act."Thickness"
    AND cm."Length" = act."Length"
LEFT JOIN IdealSegments ideal
    ON cm."Location" = ideal."Location"
    AND cm."DF_Market" = ideal."DF_Market"
    AND cm."Flavor" = ideal."Flavor"
    AND cm."Taste" = ideal."Taste"
    AND cm."Thickness" = ideal."Thickness"
    AND cm."Length" = ideal."Length";


select * from PARIS_Output


-- Verify the table creation


-- Step 22: Create ClusterList table for Category D
CREATE OR REPLACE TEMPORARY TABLE ClusterList AS
SELECT
  m."Location",
  m."DF_Market",
  m."TMO",
  m."Flavor",
  m."Taste",
  m."Thickness",
  m."Length",
  COUNT(DISTINCT m."SKU") AS "PMI SKU",
  CAST(0 AS INTEGER) AS "Cluster SKU"  -- Explicitly define as INTEGER
FROM MarketSummary m
WHERE m."TMO" = 'PMI'
GROUP BY m."Location", m."DF_Market", m."TMO", m."Flavor", m."Taste", m."Thickness", m."Length";


select * from ClusterList

-- Update ClusterList with SKU counts from other locations with the same DF_Market
UPDATE ClusterList c
SET "Cluster SKU" = CAST((
  SELECT COUNT(DISTINCT m."SKU")
  FROM MarketSummary m
  WHERE m."DF_Market" = c."DF_Market"    -- Same market
    AND m."Location" != c."Location"     -- Different location
    AND m."Flavor" = c."Flavor"
    AND m."Taste" = c."Taste"
    AND m."Thickness" = c."Thickness"
    AND m."Length" = c."Length"
    AND m."TMO" = 'PMI'
) AS INTEGER);


select * from ClusterList

-- Step 23: Calculate Category scores

-- Category A Scores
CREATE OR REPLACE TEMPORARY TABLE CategoryAScores AS
SELECT
  loc."Location",
  ((green."GreenCount" - (red."RedCount" * 2)) / loc."TotalSKU") * 100 AS "Score_A",
  ((((green."GreenCount" - (red."RedCount" * 2)) / loc."TotalSKU") * 100) + 200) * (10 / 300) AS "ScaledScore_A"
FROM (
  SELECT
    "Location",
    COUNT(*) AS "TotalSKU"
  FROM df_vols_w_metrics
  WHERE "TMO" = 'PMI'
  GROUP BY "Location"
) loc
LEFT JOIN (
  SELECT
    "Location",
    COUNT(*) AS "GreenCount"
  FROM Flags
  WHERE "FlagType" = 'Green'
  GROUP BY "Location"
) green ON loc."Location" = green."Location"
LEFT JOIN (
  SELECT
    "Location",
    COUNT(*) AS "RedCount"
  FROM Flags
  WHERE "FlagType" = 'Red'
  GROUP BY "Location"
) red ON loc."Location" = red."Location";


select * from CategoryAScores


-- Category B Scores
CREATE OR REPLACE TEMPORARY TABLE CategoryBScores AS
SELECT
  loc."Location",
  CAST(REGR_R2(comp."SoM_Comp", pmi."SoM_PMI") * 10 AS FLOAT) AS "Cat_B",
  MAX(pmi."PMI Total") AS "NumPMI_SKU",
  MAX(comp."Comp_Total") AS "NumComp_SKU",
  SUM(CASE WHEN m."TMO" = 'PMI' THEN m."$current_year_Volume" ELSE 0 END) AS "PMI_Volume",
  SUM(CASE WHEN m."TMO" != 'PMI' THEN m."$current_year_Volume" ELSE 0 END) AS "Comp_Volume"
FROM (SELECT DISTINCT "Location" FROM MarketSummary) loc
LEFT JOIN Market_Summary_PMI pmi ON loc."Location" = pmi."Location"
LEFT JOIN Market_Summary_Comp comp
  ON pmi."Location" = comp."Location"
  AND pmi."Flavor" = comp."Flavor"
  AND pmi."Taste" = comp."Taste"
  AND pmi."Thickness" = comp."Thickness"
  AND pmi."Length" = comp."Length"
LEFT JOIN MarketSummary m ON loc."Location" = m."Location"
GROUP BY loc."Location";


select * from CategoryBScores


-- Category C Scores
CREATE OR REPLACE TEMPORARY TABLE CategoryCScores AS
WITH AllLocations AS (
  SELECT DISTINCT "Location" FROM cat_a_df_vols
),
PARISScores AS (
  SELECT
    p."Location",
    CAST(REGR_R2(p."Real_So_Segment", p."Ideal_So_Segment") * 10 AS FLOAT) AS "PARIS_Score"
  FROM PARIS_Output p
  GROUP BY p."Location"
  HAVING COUNT(*) >= 2  -- Need at least 2 data points for regression
),
MarketScores AS (
  SELECT
    c."DF_Market",
    AVG(ps."PARIS_Score") AS "Market_Score"
  FROM cat_a_df_vols c
  JOIN PARISScores ps ON c."Location" = ps."Location"
  GROUP BY c."DF_Market"
),
LocationWithMarketScores AS (
  SELECT
    c."Location",
    COALESCE(ms."Market_Score", 5.0) AS "Market_Score"
  FROM cat_a_df_vols c
  LEFT JOIN MarketScores ms ON c."DF_Market" = ms."DF_Market"
)
SELECT
  al."Location",
  COALESCE(ps."PARIS_Score", lms."Market_Score", 5.0) AS "Cat_C"
FROM AllLocations al
LEFT JOIN PARISScores ps ON al."Location" = ps."Location"
LEFT JOIN LocationWithMarketScores lms ON al."Location" = lms."Location";


select * from CategoryCScores

-- Category D Scores
CREATE OR REPLACE TEMPORARY TABLE CategoryDScores AS
SELECT
  "Location",
  CASE
    WHEN SUM("PMI SKU") > 0 THEN CAST((SUM("Cluster SKU") / SUM("PMI SKU")) AS FLOAT)
    ELSE 0
  END AS "Cat_D"
FROM ClusterList
GROUP BY "Location";


select * from CategoryDScores


-- Step 24: Calculate Location Volumes
CREATE OR REPLACE TEMPORARY TABLE LocationVolumes AS
SELECT
  "Location",
  SUM("$current_year Volume") AS "Market_Volume"
FROM cat_a_df_vols
GROUP BY "Location";



select * from LocationVolumes





CREATE OR REPLACE TEMPORARY TABLE PMIVolumes AS
SELECT
  "Location",
  SUM("$current_year Volume") AS "PMI_Volume"
FROM cat_a_df_vols
WHERE "TMO" = 'PMI'
GROUP BY "Location";

select * from PMIVolumes


-- Step 25: Final score calculation
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
  (COALESCE(a."ScaledScore_A", 0) + COALESCE(b."Cat_B", 0) +
   COALESCE(c."Cat_C", 0) + COALESCE(d."Cat_D", 0)) / 4 AS "Avg_Score",
  lv."Market_Volume",
  pv."PMI_Volume",
  CASE
    WHEN COALESCE(lv."Market_Volume", 0) > 0 THEN COALESCE(pv."PMI_Volume", 0) / lv."Market_Volume"
    ELSE 0
  END AS "Market_Share"
FROM AllLocations al
LEFT JOIN CategoryAScores a ON al."Location" = a."Location"
LEFT JOIN CategoryBScores b ON al."Location" = b."Location"
LEFT JOIN CategoryCScores c ON al."Location" = c."Location"
LEFT JOIN CategoryDScores d ON al."Location" = d."Location"
LEFT JOIN LocationVolumes lv ON al."Location" = lv."Location"
LEFT JOIN PMIVolumes pv ON al."Location" = pv."Location"
ORDER BY "Avg_Score" DESC;


select * from FinalScores

-- View the final results