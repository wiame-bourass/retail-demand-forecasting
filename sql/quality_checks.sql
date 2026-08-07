-- Key uniqueness.
SELECT date, store_nbr, family, COUNT(*) AS n
FROM read_parquet('data/processed/curated_panel.parquet')
GROUP BY date, store_nbr, family
HAVING COUNT(*) > 1;

-- Invalid targets.
SELECT
    SUM(CASE WHEN sales < 0 THEN 1 ELSE 0 END) AS negative_sales_rows,
    SUM(CASE WHEN NOT is_test AND sales IS NULL THEN 1 ELSE 0 END) AS missing_historical_targets,
    SUM(CASE WHEN is_test AND sales IS NOT NULL THEN 1 ELSE 0 END) AS future_target_leakage_rows
FROM read_parquet('data/processed/curated_panel.parquet');

-- Promotion consistency.
SELECT *
FROM read_parquet('data/processed/curated_panel.parquet')
WHERE n_items_on_promotion > catalog_items_family
   OR promotion_share_catalog < 0
   OR promotion_share_catalog > 1;
