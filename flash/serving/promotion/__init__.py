"""Deploy-time promotion gate for the hosted serving release.

Imported by `.github/workflows/deploy-modal.yml` AFTER `modal deploy`, never by the request path.
Nothing here may be imported from `flash/serving/src/http/` or `flash/serving/src/engine/`: this
code runs on a CI runner against a live deployment, not inside the serving container.

`/healthz` proves a router process answered with the identity the deploy step injected. It cannot
prove a GPU engine started, that a generation ran, that streaming works, or that usage settled
durably. This package supplies that missing evidence and fails closed without it.
"""
