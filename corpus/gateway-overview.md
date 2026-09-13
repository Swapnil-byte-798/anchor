# Meridian Gateway Overview

Meridian is the internal inference platform that every product team at the
company builds on. The gateway is its single front door: a request is
authenticated, metered against the tenant quotas defined in
rate-limits-and-quotas, routed to a model pool, and then logged under the
schedule set out in data-retention-policy. This page is the orientation
document for anyone joining the Platform Gateway team, and the rest of the
handbook assumes the vocabulary defined here.

## Public endpoints

The gateway exposes four public endpoints and one administrative endpoint. The
/v1/generate endpoint serves streaming and non-streaming completions and carries
78% of all production traffic. The /v1/embed endpoint returns 1024-dimensional
vectors and is called mostly in batch by the overnight indexing job. The
/v1/rerank endpoint accepts up to 100 candidate passages in a single call and is
used only by the search product. The /v1/status endpoint is unauthenticated and
returns the build identifier together with a pool health summary. The
/v1/admin/reindex endpoint is restricted to the Platform Gateway team and
rebuilds one tenant index in place.

## Latency budgets

Every endpoint has a published p95 latency budget that the gateway steward
reviews each quarter. The published p95 budget for /v1/generate is 850 ms
measured at the edge. The published p95 budget for /v1/embed is 120 ms. The
published p95 budget for /v1/rerank is 310 ms. The published p95 budget for
/v1/status is 40 ms. A budget breach that persists longer than 15 minutes opens
an incident at the severity given by incident-severity, and the on-call response
is the one described in oncall-runbook.

## Regions and failover

Meridian serves traffic from three regions: us-east-1 in Ashburn, eu-west-1 in
Dublin, and ap-south-1 in Mumbai. Pool health checks run every 10 seconds and a
pool is drained after three consecutive failures. A full regional failover
completes in 90 seconds and is exercised in a game day once per quarter.
Cross-region replication of a tenant index lags by 45 seconds under normal load,
and that lag is the reason a reindex is never considered complete until the
follower regions have caught up.

## Authentication and tenancy

Every call carries a tenant key issued by the platform identity service and
scoped to a single region. A tenant key expires after 180 days and the rotation
reminder is sent 14 days before expiry. A call presenting an expired key is
rejected with a 401 and is not counted against the tenant quota. The gateway
allows 50 concurrent connections per tenant key by default, and the Platform
Gateway team raises that ceiling on request. Tenant identifiers follow the
pattern tnt-0471 and appear on every log line the gateway writes.

## Ownership and support

The Platform Gateway team owns the gateway, the model pools and the index build
pipeline. The gateway steward is the named engineer accountable for the latency
budgets and for the weekly capacity review. Product teams reach the team in the
#meridian-gateway channel during business hours and through the on-call rotation
at all other times. Platform availability is reported against a 99.95% monthly
target, and the trailing monthly series is published in metrics.csv.
