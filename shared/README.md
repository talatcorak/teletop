# shared

Cross-cutting notes shared between `server`, `web`, and `client`.

Use this directory for things like:

- WebSocket message schema documentation
- HTTP API contracts (until/if we generate them from FastAPI's OpenAPI)
- Build-artifact / manifest format
- Wire-format constants

Code that needs to be imported should live in its consumer (e.g. server or
client) rather than here, since this is not a build target.
