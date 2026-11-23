# runtime-selfcheck

## Purpose

Defines how the system verifies, before serving traffic, that its external model dependencies are reachable and behave as configured, so that misconfiguration is reported as a configuration problem instead of surfacing later as an opaque failure.

## Requirements

### Requirement: Provider verification on startup

The system SHALL verify each external model dependency it depends on and SHALL report per-dependency status, distinguishing reachable-and-configured from misconfigured or unreachable.

#### Scenario: All providers healthy
- **WHEN** the service starts with valid configuration and reachable providers
- **THEN** each dependency is reported as healthy

#### Scenario: One provider misconfigured
- **WHEN** a provider is unreachable or rejects the configured credentials or model
- **THEN** that dependency is reported as unhealthy with a message identifying the provider and the configuration value implicated

#### Scenario: Startup does not depend on full availability
- **WHEN** one or more providers are unhealthy at startup
- **THEN** the system still starts and reports the affected capabilities as unavailable rather than refusing to run entirely

### Requirement: Embedding dimension discovery

The system SHALL determine the embedding vector dimension from the configured embedding provider rather than assuming a fixed dimension, and SHALL use the discovered dimension when initializing storage.

#### Scenario: Dimension is discovered
- **WHEN** embedding verification succeeds
- **THEN** the observed vector dimension is reported and used as the dimension of stored vectors

#### Scenario: Configured dimension conflicts with provider
- **WHEN** a dimension is configured that disagrees with the dimension the provider returns
- **THEN** the system reports the conflict explicitly, naming both values

#### Scenario: Unknown embedding model
- **WHEN** the embedding provider rejects the configured model name
- **THEN** the system reports that the model is unknown or unavailable, rather than reporting a generic request failure

### Requirement: Stored dimension consistency

The system SHALL record the embedding model identity and dimension used to build existing storage, and SHALL refuse to silently mix vectors of differing dimensions or differing models within the same collection.

#### Scenario: Model changed after data exists
- **WHEN** the configured embedding model or dimension differs from what existing stored data was built with
- **THEN** the system reports the mismatch and requires an explicit rebuild rather than producing mixed or invalid results

#### Scenario: Fresh store adopts current configuration
- **WHEN** no stored data exists
- **THEN** the system initializes storage using the currently discovered model and dimension

### Requirement: Reranker verification

The system SHALL verify the reranking dependency using a representative request and SHALL confirm the response has the expected shape and produces a meaningful ordering, not merely that the endpoint responded.

#### Scenario: Reranker verified with representative input
- **WHEN** reranker verification runs
- **THEN** a small request with a clearly most-relevant document is issued and the returned ordering is checked for expected shape and sensible ranking

#### Scenario: Reranker returns unexpected shape
- **WHEN** the reranker responds with a structure that does not match expectations
- **THEN** the system reports a contract mismatch rather than using the response

### Requirement: Generation provider verification

The system SHALL verify that the configured generation provider and model are usable, including that responses can be delivered incrementally.

#### Scenario: Generation model verified
- **WHEN** generation verification runs
- **THEN** a minimal request is issued and a response confirms the model exists and credentials are accepted

#### Scenario: Streaming capability verified
- **WHEN** generation verification runs
- **THEN** incremental delivery is exercised so that a non-streaming provider is detected before it affects users

### Requirement: Health endpoint

The system SHALL expose the results of dependency verification so that an operator or the interface can determine current dependency state without reading logs.

#### Scenario: Health is queryable
- **WHEN** the health endpoint is requested
- **THEN** per-dependency status is returned, distinguishing healthy, misconfigured, and unreachable

#### Scenario: Health reflects configuration errors precisely
- **WHEN** a dependency is misconfigured
- **THEN** the returned status identifies the specific misconfiguration such as an absent credential or unknown model
