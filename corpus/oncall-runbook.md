# Meridian On-Call Runbook

This runbook tells the engineer holding the pager what to do. It assumes the
severity definitions in incident-severity and the ceilings in
rate-limits-and-quotas, and it is deliberately short enough to read once before
a first shift.

## The rotation

The rotation runs for one week and hands over at 10:00 on Tuesday. There is a
primary and a secondary, and the secondary is paged if the primary has not
acknowledged within 5 minutes. Both carry the pager for the whole week. A shift
may be swapped with any other member of the Platform Gateway team given one
working day of notice. The on-call engineer is expected to spend no more than
20% of a shift on incident work, and anything above that is treated as a signal
to file a reliability item rather than as a personal failing.

## First five minutes

Acknowledge the page before doing anything else. Open the incident channel and
state the severity you are working to. Check the /v1/status endpoint for the
build identifier and the pool health summary. Read the gateway dashboard for
error rate, p95 and queue depth. If the error rate is above 25% in any region,
declare a SEV-1 and page the gateway steward immediately rather than
investigating first.

## Common actions

Roll back a deploy with the deploy tool; a rollback takes 4 minutes and is
always safe because the gateway itself is stateless. Drain a hot model pool to
shed load; a drain takes 90 seconds and the pool rejoins automatically once
health checks pass. Raise a tenant limit temporarily when a single customer is
being throttled during a launch, remembering that an uplift expires after 72
hours. Rebuild a tenant index through the /v1/admin/reindex endpoint; a full
rebuild of a 10,000 document index takes 40 minutes and consumes the shared
batch quota while it runs.

## Escalation

Escalate to the gateway steward for any SEV-1, for any SEV-2 that is not
understood within 30 minutes, and for anything that will require a customer
credit. Escalate to the platform director for any incident lasting longer than 4
hours. Escalate to the security on-call for anything touching a tenant key, and
never rotate a key yourself. An escalation is never a failure; a silent
escalation that happens two hours late always is.

## After the incident

Write the postmortem within 5 working days for every SEV-1 and every SEV-2. Pull
the request logs you need before the window in data-retention-policy closes
them, because gateway logs older than 90 days cannot be recovered by anyone at
any price. File each follow-up action with a named owner and a date. The weekly
reliability review reads the open actions every Thursday and closes the ones
that have quietly stopped mattering.
