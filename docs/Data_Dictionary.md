# Data Dictionary — `data/students.csv`

| Column | Type | Range / values | Description |
|---|---|---|---|
| student_id | string | S100000… | Unique synthetic identifier |
| gender | categorical | Female, Male | Self-reported gender |
| age | int | 17–23 | Age in years |
| parental_education | ordinal | High School < Some College < Bachelor < Master < Doctorate | Highest parental education (≈2% missing) |
| internet_access | binary | Yes / No | Reliable home internet access |
| part_time_job | binary | Yes / No | Works during the semester |
| extracurricular | binary | Yes / No | Takes part in clubs or sports |
| tutoring | binary | Yes / No | Enrolled in tutoring |
| study_hours_weekly | float | 0–35 | Self-reported weekly study hours |
| attendance_pct | float | 40–100 | Class attendance percentage by mid-semester |
| sleep_hours | float | 4–10 | Average nightly sleep (≈2% missing) |
| previous_gpa | float | 1.0–4.0 | Cumulative GPA before this course |
| past_failures | int | 0–3 | Number of previously failed courses |
| lms_logins_monthly | int | ≥ 0 | Learning-management-system logins per month (≈2% missing) |
| assignment_completion_pct | float | 20–100 | Share of assignments submitted |
| midterm_score | float | 0–100 | Midterm exam score |
| **final_score** | float | 0–100 | **Regression target**: final course score |
| **at_risk** | int | 0 / 1 | **Classification target**: 1 if final_score < 60 |
| grade | categorical | A–F | Letter grade derived from final_score (not used as a feature) |

## Engineered feature

| Feature | Formula |
|---|---|
| engagement_index | 0.4 × attendance/100 + 0.3 × assignment_completion/100 + 0.3 × (lms_logins / max lms_logins) |

## Leakage guard

`final_score`, `at_risk` and `grade` are outcome columns and are never used as model inputs (see `DROP` in `src/features.py`).
