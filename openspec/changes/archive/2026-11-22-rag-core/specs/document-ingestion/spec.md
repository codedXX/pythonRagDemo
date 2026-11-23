## Purpose

Defines how source documents are accepted, parsed, split into retrievable units, vectorized, and persisted, so that uploaded knowledge becomes searchable as a coherent and atomic unit.

## ADDED Requirements

### Requirement: Supported document types

The system SHALL accept PDF and Markdown documents for ingestion and MUST reject any other file type with an explicit, user-visible error naming the unsupported type.

#### Scenario: Accepted types are ingested
- **WHEN** a user uploads a file whose extension is `.pdf`, `.md`, or `.markdown` (case-insensitive)
- **THEN** the system accepts the file and starts the ingestion pipeline

#### Scenario: Unsupported type is rejected
- **WHEN** a user uploads a file with any other extension
- **THEN** the system rejects it before parsing and returns an error identifying the rejected type and listing the supported types

### Requirement: Scanned PDF detection

The system SHALL detect PDFs that yield little or no extractable text and MUST refuse to ingest them, reporting that the document appears to be image-based, rather than ingesting empty or meaningless content.

#### Scenario: Image-only PDF is refused
- **WHEN** a PDF's extractable text density falls below the configured minimum threshold
- **THEN** the system aborts ingestion for that document and reports that it appears to be a scanned or image-based PDF requiring OCR

#### Scenario: Text PDF proceeds normally
- **WHEN** a PDF's extractable text density is at or above the configured threshold
- **THEN** the system proceeds with normal parsing

### Requirement: Structural metadata preservation

For every accepted document, the system SHALL preserve provenance metadata sufficient to cite the source, including the document name, and — where available — section headings and page numbers.

#### Scenario: Metadata survives to retrieval
- **WHEN** a passage from an ingested document is returned to a user
- **THEN** the citation includes the source document name and, when available, the page number and section heading path for that passage

### Requirement: Two-level parent-child chunking

The system SHALL split documents into larger parent units and smaller child units, where every child unit belongs to exactly one parent unit, and SHALL retain the parent text so it can be expanded at retrieval time.

#### Scenario: Every child has a resolvable parent
- **WHEN** a document finishes chunking
- **THEN** every child unit references a parent unit that exists and whose text is stored

#### Scenario: Parent units respect a size ceiling
- **WHEN** a candidate parent unit would exceed the configured maximum size
- **THEN** the system splits it at the next available boundary so that no parent unit exceeds the configured maximum

### Requirement: Semantic boundary preference

The system SHALL prefer structural boundaries (headings, layout blocks, paragraphs) when splitting, and SHALL apply semantic boundary detection only within segments that remain oversized after structural splitting.

#### Scenario: Structurally bounded documents skip semantic splitting
- **WHEN** all structural segments of a document are within the configured size limit
- **THEN** no semantic boundary detection is performed for that document

#### Scenario: Oversized segments fall back to semantic detection
- **WHEN** a structural segment exceeds the configured size limit
- **THEN** the system applies semantic boundary detection within that segment to choose split points

### Requirement: Embedding calls are resilient

The system SHALL batch embedding requests within a configurable batch size and MUST retry transient failures with backoff rather than failing the entire ingestion on the first error.

#### Scenario: Transient failure is retried
- **WHEN** an embedding request fails with a retryable error
- **THEN** the system retries with increasing delay up to a configured attempt limit before reporting failure

#### Scenario: Non-retryable failure surfaces clearly
- **WHEN** an embedding request fails with a non-retryable error such as invalid credentials or an unknown model
- **THEN** the system stops retrying and reports the underlying cause in terms of configuration rather than an opaque transport error

### Requirement: Embedding reuse by content hash

The system SHALL cache embeddings keyed by a hash of the embedded content and embedding model, so that re-ingesting unchanged text does not re-invoke the embedding service.

#### Scenario: Unchanged text is not re-embedded
- **WHEN** text whose content hash and model match a cached entry is submitted for embedding
- **THEN** the cached vector is reused and no embedding request is issued for that text

#### Scenario: Changed text is re-embedded
- **WHEN** text content or the configured embedding model differs from any cached entry
- **THEN** a fresh embedding request is issued and the result is cached

### Requirement: Atomic ingestion

Ingestion of a document SHALL be atomic: content written during a failed or interrupted ingestion MUST NOT become retrievable, and partially ingested documents MUST NOT be visible to retrieval.

#### Scenario: Failed ingestion leaves nothing retrievable
- **WHEN** ingestion fails or is interrupted before completion
- **THEN** none of that document's units are returned by any subsequent retrieval

#### Scenario: Successful ingestion becomes retrievable immediately
- **WHEN** ingestion completes successfully
- **THEN** the document is immediately retrievable, without requiring a restart or a manual visibility step

### Requirement: Document lifecycle management

The system SHALL maintain a registry of ingested documents with their state and unit counts, and SHALL support listing available documents and removing a document along with all of its retrievable content.

#### Scenario: Listing documents
- **WHEN** a user requests the document list
- **THEN** the system returns each ingested document with its name, state, and unit counts

#### Scenario: Removing a document
- **WHEN** a user removes an ingested document
- **THEN** all of its parent and child units cease to be retrievable and it no longer appears in the document list

#### Scenario: Re-uploading the same document
- **WHEN** a document with the same name is ingested again
- **THEN** the system does not leave duplicate retrievable copies of its units

### Requirement: Ingestion progress reporting

The system SHALL expose ingestion progress through distinct stages so that a user can distinguish active processing from a stalled or failed operation.

#### Scenario: Stage transitions are observable
- **WHEN** a document is being ingested
- **THEN** the user can observe progression through parsing, chunking, embedding, and persistence stages

#### Scenario: Failure is reported at its stage
- **WHEN** ingestion fails
- **THEN** the report identifies the stage at which it failed together with an actionable message