# Meridian Data Retention Policy

This policy states how long Meridian keeps each class of data, who may delete
it, and what the platform team must do before a window closes. It is the reason
several investigations described elsewhere in this handbook carry a hard
deadline, and it is the single most quoted page in the set.

## Retention windows

Gateway request logs are retained for 90 days and are then deleted irreversibly.
Prompt and completion bodies are retained for 30 days and are stored apart from
the request logs. The embedding cache is retained for 30 days after its last
read. Aggregate metrics, including the monthly series published in metrics.csv,
are retained for 400 days. Billing records are retained for seven years to
satisfy the finance team's audit obligation.

## Why ninety days

The 90-day window on request logs is a deliberate trade-off between
investigation depth and storage cost. Archive storage is billed at $180 per
month per region across the three Meridian regions. Extending request logs to
180 days was estimated at $7,900 per year once index and egress charges were
counted, and the platform team accepted the shorter window in the 2024 review.
The consequence is written down plainly in postmortem-2025-06-index-drift: a
slow regression that starts outside the window cannot be reconstructed from
request logs at all.

## Deletion requests

A tenant deletion request is acknowledged within 24 hours and completed within
30 days. Deletion covers request logs, prompt bodies, the embedding cache and
any tenant index held in any region. Billing records are exempt from deletion
and are held for the full seven years. The data protection lead signs off every
deletion request, and the deletion register is retained for 400 days.

## Access

Request logs are readable by the Platform Gateway team and by the security
on-call. Prompt and completion bodies are readable only by the data protection
lead, and only against a written reason recorded in the access register. Every
read of a prompt body is itself logged, and those access records are retained
for 400 days. No access path exists that leaves no trace.

## Backups

Overnight backups of the index database are kept for 14 days. A backup restore
has been exercised twice and takes 25 minutes for a full region. Backups are
encrypted at rest and the encryption key is rotated every 180 days. A restore
does not resurrect request logs that the retention job has already deleted,
which surprises people often enough to be worth writing down.

## Review

This policy is reviewed every 12 months by the data protection lead together
with the gateway steward. The next review is due in March 2026. A change to any
window requires a written impact note covering incident investigation, billing
reconciliation and the index rebuild schedule.
