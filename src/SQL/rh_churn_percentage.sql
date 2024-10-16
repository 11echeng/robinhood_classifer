-- Creating a new column streak_id as a way to identify consecutive days. Note: This assumes no duplication in dates per user. Need to validate.
WITH temp1 AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS rn,
        timestamp::date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp)::int AS streak_id
    FROM
        equity_value_data
),
-- Creating the start and end dates for each conseuctive streak of $10 for each user and how long each streak is.
temp2 AS (
    SELECT
        user_id,
        MIN(timestamp::date) AS start_streak_date,
        MAX(timestamp::date) AS end_streak_date,
        COUNT(*) AS duration_of_above10_streak
    FROM
        temp1
    GROUP BY
        user_id, streak_id
),
-- Creating the last end date range of consecutive streaks for the start of each consecutive streak.
-- Brainstorming: I'm wondering for users who stared with less than 10$, using that date and the first time they had a streak of $10+, which wuould be the minimum, and finding the day difference??????
temp3 AS (
    SELECT
        *,
        LAG(end_streak_date) OVER (PARTITION BY user_id ORDER BY start_streak_date ASC) AS prev_above10_streak_date,
        start_streak_date - LAG(end_streak_date) OVER (PARTITION BY user_id ORDER BY start_streak_date ASC) AS duration_between_above10_streaks
    FROM
        temp2
),
-- Filtering for users who had 28+ gap days between the end of a user's $10 streak and the start of a user's $10 streak.
temp4 AS (
    SELECT *
    FROM temp3
    WHERE duration_between_above10_streaks >= 28
)
SELECT
    CONCAT(
        ROUND(
            (COUNT(DISTINCT user_id) * 1.0 / (SELECT COUNT(DISTINCT user_id) FROM features_data)) * 100, 3
        ), '%'
    ) AS churn_rate
FROM
    temp4;