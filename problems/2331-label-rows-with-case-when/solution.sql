-- Readings per temperature band
SELECT
    CASE
        WHEN temperature < 10 THEN 'cold'
        WHEN temperature < 25 THEN 'mild'
        ELSE 'hot'
    END AS band,
    COUNT(*) AS reading_count
FROM readings
GROUP BY
    CASE
        WHEN temperature < 10 THEN 'cold'
        WHEN temperature < 25 THEN 'mild'
        ELSE 'hot'
    END;
