# Postmortem: June 2025 Index Drift

## Summary

Between 2025-04-18 and 2025-06-11 the search index for 12 Scale tier tenants
drifted out of date. A document written after 2025-04-18 was accepted by the
gateway but never became searchable. The incident was opened as a SEV-2 on
2025-06-11 and was closed on 2025-06-13. Customer impact was a silent recall
loss: no call failed, no latency budget was breached, and consequently nothing
paged anybody for 54 days.

## Timeline

On 2025-04-18 a configuration change moved the overnight reindex job onto the
shared batch quota described in rate-limits-and-quotas. The job began to exceed
the 250,000 requests per hour batch quota on its very first run and was queued
rather than rejected. The queued job then ran for six weeks without ever
completing a full pass. On 2025-06-11 a customer reported that a document
uploaded in May was still not searchable. The on-call engineer confirmed the
drift within 40 minutes and declared a SEV-2 under the automatic rules in
incident-severity.

## What went wrong

Three things had to be true at the same time. The batch quota rejects nothing,
so exceeding it queues the job and emits a warning that no alert was consuming.
The reindex job reported success on a partial pass, so the freshness signal the
team believed in never dropped. The request logs that would have shown the
stalled pattern in April were already deleted by the time anyone went looking
for them. That third condition is a direct consequence of data-retention-policy,
which keeps gateway request logs for 90 days and no longer.

## Why it went unnoticed

The honest answer is that nothing in the platform measured index freshness.
Availability stayed above the 99.95% monthly target for the entire period. The
gateway p95 never rose above 806 ms during the drift, comfortably inside the 850
ms budget published for /v1/generate. By December 2025 the gateway p95 had
settled at 690 ms. June 2025 closed with four recorded incidents, the highest
count of any month that year.

## Actions

Alert on batch quota queue depth above 1,000 jobs, owned by the gateway steward
and due 2025-07-01. Emit an index freshness metric for every tenant and open a
SEV-3 when an index has not been rebuilt for 30 days, a rule that now appears in
incident-severity. Make the reindex job fail loudly on a partial pass instead of
reporting success. Extend request log retention for the batch pipeline alone to
180 days at an estimated $1,200 per year, approved in the July review.

## What we are not doing

We are not extending request log retention across the whole platform, because
the cost case in data-retention-policy has not changed. We are not adding a
second index backend. The team judged that a freshness metric and a loud failure
remove the entire class of problem for far less money.
