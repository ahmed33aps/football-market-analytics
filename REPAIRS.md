# Repairs and limits

- Removed dependency on unreviewed serialized Python model objects. The dashboard trains a deterministic local random forest from supplied data instead of loading uploaded pickle files.
- Rebuilt categorical encoders and scalers in notebook execution order.
- Kept the original genetic-algorithm implementation for study, but the default run uses the dashboard's fixed 12-feature set. No fresh GA-selected score is claimed.
- Dashboard inputs are now trained in the same raw units used by the form; earlier saved model preprocessing was not reproducible.
- This is a descriptive/in-sample portfolio demo. Same-player repeated rows and historical peak value can cause leakage. Training fit and a dashboard output are not evidence of future transfer-market prediction accuracy.
- Original model binaries remain only in the original source archive, not in this repository.
