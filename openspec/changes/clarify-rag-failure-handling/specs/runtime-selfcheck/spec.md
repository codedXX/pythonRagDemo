## MODIFIED Requirements

### Requirement: Stored dimension consistency

The system SHALL record the embedding model identity and dimension used to build existing storage, and SHALL refuse to silently mix vectors of differing dimensions or differing models within the same collection. The health report SHALL identify storage access failures and invalid stored metadata. Automatic initialization and ingestion SHALL protect existing data whenever compatibility cannot be established.

#### Scenario: Model changed after data exists
- **WHEN** the configured embedding model or dimension differs from what existing stored data was built with
- **THEN** the system reports the mismatch and requires an explicit rebuild rather than producing mixed or invalid results

#### Scenario: Fresh store adopts current configuration
- **WHEN** no stored data exists
- **THEN** the system initializes storage using the currently discovered model and dimension

#### Scenario: Metadata query fails
- **WHEN** a storage operation fails while reading metadata or checking whether passages exist
- **THEN** the schema check reports a storage access failure and automatic metadata writes and ingestion are blocked

#### Scenario: Existing metadata is damaged or missing
- **WHEN** a stored schema record is invalid or existing passages lack a schema record
- **THEN** the schema check reports the metadata problem and the original records are preserved

#### Scenario: Incompatible storage remains diagnosable
- **WHEN** startup or the self-check command detects an incompatible model or dimension
- **THEN** the original schema remains available for diagnosis, the web service can report the failure, and ingestion cannot replace existing passages
