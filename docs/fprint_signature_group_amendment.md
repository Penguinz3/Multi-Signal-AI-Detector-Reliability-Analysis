# Signature-group allocation amendment

Date: 2026-08-22

The first zero-score forecast preflight stopped before producing an artifact or
lock because the PMC signature partition had 1,008 records but only 215 article
groups. A subsequent count found the same preflight failure condition for
Gutenberg (1,011 records, 108 author groups). No target detector scores,
forecasts, forecast locks, or target outcomes existed when this amendment was
made.

The generic grouped allocator previously filled the signature partition using
only a 1,000-record quota. It now assigns whole groups until the signature has
both at least 1,000 records and at least 250 groups, then assigns at least 2,000
test records. Author/user/article grouping, prompt stratification, the 20 seeded
nested draws at sizes 50/100/250, and every target-score embargo remain
unchanged. The earlier prepared/scored state is retained only as a failed
preflight checkpoint; the experiment is rebuilt before forecasting.
