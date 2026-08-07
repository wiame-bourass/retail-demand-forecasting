-- DuckDB exploratory queries on the curated panel.

-- 1. Volume and promotion rate by family.
SELECT
    family,
    SUM(sales) AS total_sales,
    AVG(CAST(any_promotion AS INTEGER)) AS promotion_rate,
    AVG(sales) AS mean_daily_store_sales
FROM read_parquet('data/processed/curated_panel.parquet')
WHERE NOT is_test
GROUP BY family
ORDER BY total_sales DESC;

-- 2. Weekly seasonality by store and family.
SELECT
    store_nbr,
    family,
    dayofweek(date) AS day_of_week,
    AVG(sales) AS mean_sales,
    MEDIAN(sales) AS median_sales
FROM read_parquet('data/processed/curated_panel.parquet')
WHERE NOT is_test
GROUP BY store_nbr, family, day_of_week
ORDER BY store_nbr, family, day_of_week;

-- 3. Promotion association, explicitly descriptive and not causal.
SELECT
    family,
    any_promotion,
    COUNT(*) AS observations,
    AVG(sales) AS mean_sales,
    MEDIAN(sales) AS median_sales
FROM read_parquet('data/processed/curated_panel.parquet')
WHERE NOT is_test
GROUP BY family, any_promotion
ORDER BY family, any_promotion;

-- 4. Bottom-up aggregation levels.
SELECT date, state, city, store_nbr, family, SUM(sales) AS sales
FROM read_parquet('data/processed/curated_panel.parquet')
WHERE NOT is_test
GROUP BY date, state, city, store_nbr, family;
