# AI Usage Log

Fill this in continuously as you go — it directly feeds the "Responsible Use of GenAI" grading criteria and the Academic Integrity Acknowledgement appendix. One entry per meaningful AI interaction.

| Date | Tool | Student | Prompt | Output Summary | Adopted / Rejected | Why |
|---|---|---|---|---|---|---|
| 2026-07-09 | Claude Code | (team) | Recommend a project scenario + dataset with broad, low-error-risk data for the ET0737 mini project | Recommended Customer Churn Prediction (classification) using the IBM Telco Customer Churn dataset | Adopted | Dataset has genuine mixed numeric/categorical features (forces real preprocessing work), workable class balance (~26% churn, no heavy resampling needed), matches the module's suggested classification algorithms, and supports clear business insights and an easy interactive demo |
| 2026-07-09 | Claude Code | (team) | Verify our dataset is the best-fit, canonical, real (non-AI-generated) version for churn prediction and confirm the citation source | Web search confirmed our file (7,043 rows x 21 cols) matches the most-used Kaggle version exactly (blastchar/telco-customer-churn), originally IBM Cognos Analytics sample data. Local integrity check: 0 duplicate customer IDs, only 11 blank TotalCharges values (tenure=0 customers), 73.5%/26.5% class balance. | Adopted (kept existing dataset, no swap needed) | Confirmed via independent web search + local stdlib verification, not just AI assertion; Kaggle URL identified as the standard citation for the project PPT |

