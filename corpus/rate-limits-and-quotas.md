# Meridian Rate Limits and Quotas

This page defines the limits the gateway enforces, what a tenant sees when it
crosses one, and what an overage costs. The limits apply to the endpoints listed
in gateway-overview and are evaluated before any model pool is touched, so a
throttled call never consumes inference capacity.

## Tiers and sustained limits

Meridian sells three tiers. Starter tier tenants are limited to 600 requests per
minute. Scale tier tenants are limited to 5,000 requests per minute. Enterprise
tier tenants are limited to 20,000 requests per minute and are additionally
capped at 2,000,000 requests per day. A limit is counted per tenant key and per
region, so an Enterprise customer holding keys in two regions has two
independent budgets and neither one borrows from the other.

## Bursts and the token bucket

Each tier may burst to twice its sustained limit for up to 30 seconds. The
bucket refills at the sustained rate and is never allowed to hold more than 60
seconds of credit. A tenant that empties the bucket receives a 429 response
carrying a Retry-After header. The gateway sets Retry-After to 5 seconds for
Starter, to 2 seconds for Scale and to 1 second for Enterprise, because the
recovery time a tenant can tolerate scales with the tier it pays for.

## Overage and billing

A tenant may exceed its daily cap by up to 10% before calls are rejected
outright. Traffic inside that band is billed as overage at $0.40 per 1,000
requests. Traffic above the 10% band is refused with a 429 and is never charged.
The monthly platform fee is $4,200 for Scale and $12,000 for Enterprise, while
Starter is free up to the daily cap. Billing is reconciled on the third working
day of each month against the counters published in metrics.csv.

## Batch and administrative traffic

The overnight indexing job and the /v1/admin/reindex endpoint draw on a separate
batch quota of 250,000 requests per hour shared by all tenants in a region. The
batch quota exists so that one large reindex cannot starve interactive traffic
on /v1/generate. Exceeding the batch quota does not return a 429 to the caller;
the job is queued and a warning is emitted instead. That single design choice is
the direct cause of the incident written up in postmortem-2025-06-index-drift,
where a queued reindex ran for six weeks without ever completing a pass.

## Requesting a change

A quota change is approved by the gateway steward and takes effect at the next
hourly configuration reload. A tenant may request a temporary uplift of up to 3
times its tier limit for a named launch event, and an uplift expires
automatically after 72 hours. Requests are filed in the #meridian-gateway
channel and the team answers within one working day. A permanent tier change is
a commercial decision and is routed to the account team rather than to the
Platform Gateway team.
