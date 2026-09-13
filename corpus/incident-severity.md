# Meridian Incident Severity

This page defines the severity bands used across Meridian. Every incident is
opened at a severity, and the actions that follow from it are set out in
oncall-runbook. Severity is a statement about customer impact and never a
statement about how hard the fix will be.

## The bands

SEV-1 means the gateway is unavailable in at least one region, or that more than
25% of calls are failing. SEV-2 means a published latency budget or a
correctness guarantee is being missed for a subset of tenants. SEV-3 means
degraded but usable service, such as a single model pool running hot behind a
healthy sibling. SEV-4 is a cosmetic or internal-only defect with no
customer-visible effect at all.

## Response targets

A SEV-1 must be acknowledged within 5 minutes, and a responder must be present
in the incident channel within 10 minutes. A SEV-2 must be acknowledged within
15 minutes. A SEV-3 must be acknowledged within 4 hours during business hours. A
SEV-4 is triaged in the next weekly planning session and carries no
acknowledgement target of its own.

## Communication cadence

During a SEV-1 the incident commander posts a status update every 30 minutes.
During a SEV-2 the same cadence is every 2 hours. Customer communication for a
SEV-1 or a SEV-2 is drafted by the incident commander and sent by the support
lead. Internal updates go to the #meridian-incidents channel and customer
updates go to the public status page.

## Declaring and closing

Anyone on the Platform Gateway team may declare an incident, and nobody needs
approval to raise a severity. Lowering a severity requires the agreement of both
the incident commander and the gateway steward. An incident is closed once the
triggering condition has been clear for 60 minutes. Every SEV-1 and SEV-2
requires a written postmortem within 5 working days, and
postmortem-2025-06-index-drift is the worked example new joiners are pointed at.

## Automatic severity

Some conditions open an incident without a human in the loop. Availability below
the 99.95% monthly target opens a SEV-2 at the end of the month. Sustained p95
above the published budget for 15 minutes opens a SEV-2. A failed regional
failover opens a SEV-1 immediately. A tenant index that has not been rebuilt for
30 days opens a SEV-3, a rule added after the June 2025 drift.

## Evidence and record keeping

An incident record links the alert, the deploy and the request logs that
demonstrate impact. Those logs survive only as long as data-retention-policy
allows, so an incident opened more than 90 days after the triggering change
cannot be evidenced from request logs at all. The incident register itself is
retained for 400 days, which is why the register and the logs frequently
disagree about what is still knowable.
