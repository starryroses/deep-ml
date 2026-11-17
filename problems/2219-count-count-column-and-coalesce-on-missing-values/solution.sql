SELECT
    COALESCE(department, 'Unassigned') AS department,
    COUNT(*) AS headcount,
    COUNT(bonus) AS with_bonus,
    ROUND(AVG(bonus), 2) AS avg_recorded_bonus,
    ROUND(AVG(COALESCE(bonus, 0)), 2) AS avg_bonus_all
FROM employees
GROUP BY COALESCE(department, 'Unassigned');