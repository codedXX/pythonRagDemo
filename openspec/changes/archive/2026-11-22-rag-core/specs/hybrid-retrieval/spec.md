## Purpose

Defines how a user question is turned into a ranked, provenance-rich set of passages by combining lexical and semantic recall, fusing their rankings, and reordering the result, so that answers are grounded in the most relevant available content.

## ADDED Requirements

### Requirement: Dual recall

The system SHALL retrieve candidate passages using both a lexical (keyword-based) channel and a semantic (embedding-based) channel, and MUST issue both recalls for every question.

#### Scenario: Both channels contribute
- **WHEN** a question is submitted for retrieval
- **THEN** the system produces a lexical candidate set and a semantic candidate set before any fusion

#### Scenario: One channel returns nothing
- **WHEN** either channel returns no candidates
- **THEN** the system continues with the remaining channel's results rather than failing the request

### Requirement: Chinese-aware lexical matching

The lexical channel SHALL segment Chinese text into words rather than treating text as single characters or whitespace-delimited tokens, and SHALL apply the same segmentation to both indexed content and incoming questions.

#### Scenario: Chinese question matches Chinese content
- **WHEN** a Chinese question shares salient terms with an indexed Chinese passage
- **THEN** the lexical channel ranks that passage above passages that do not share those terms

#### Scenario: Consistent segmentation
- **WHEN** text is indexed and later queried
- **THEN** both go through the same segmentation configuration, including any user-extendable dictionary and stopword list

### Requirement: Lexical index derived from the store of record

The lexical index SHALL be built from content already persisted in the vector store, and MUST be kept consistent with that store after every successful ingestion and removal.

#### Scenario: Index rebuilt after ingestion
- **WHEN** a document finishes ingesting
- **THEN** its units become lexically searchable without restarting the service

#### Scenario: Index reflects removal
- **WHEN** a document is removed
- **THEN** its units no longer appear in lexical results

#### Scenario: Index recoverable after restart
- **WHEN** the service starts and the lexical index is absent or stale
- **THEN** the system rebuilds it from persisted content rather than from original files on disk

### Requirement: Rank fusion

The system SHALL combine the lexical and semantic candidate rankings into a single ranking using rank-based fusion whose parameters are configurable, and MUST NOT depend on the raw score scales of the two channels being comparable.

#### Scenario: Fused ranking reflects both channels
- **WHEN** a passage ranks highly in both channels
- **THEN** it places above a passage that ranks highly in only one channel

#### Scenario: Fusion parameter is configurable
- **WHEN** the fusion parameter is changed in configuration
- **THEN** the fused ordering changes accordingly without code modification

### Requirement: Reranking of fused candidates

The system SHALL rerank a configurable number of top fused candidates using a dedicated reranking service, and MUST retrieve more candidates for reranking than the number ultimately supplied to generation.

#### Scenario: Rerank reorders candidates
- **WHEN** candidates are reranked
- **THEN** the final ordering follows the reranker's relevance assessment rather than the fused order

#### Scenario: Rerank candidates exceed the final supply count
- **WHEN** retrieval completes
- **THEN** the number of candidates sent to the reranker is greater than the number of passages passed to generation, so that deduplication and filtering do not exhaust the supply

#### Scenario: Reranker unavailable
- **WHEN** the reranking service fails or is unconfigured
- **THEN** the system falls back to the fused ranking and reports that reranking was skipped, rather than returning no results

### Requirement: Rerank input carries context

Passages submitted for reranking SHALL include their provenance context (for example the source name and section heading path) alongside the passage body, so that fragments whose meaning depends on their surrounding structure are scored fairly.

#### Scenario: Heading-dependent passage is scored with its heading
- **WHEN** a child passage is submitted for reranking
- **THEN** the submitted text includes that passage's heading path or document context in addition to the passage body

### Requirement: Parent expansion

The system SHALL return the parent unit's text for passages selected after reranking, so that generation receives coherent larger context rather than isolated fragments.

#### Scenario: Selected child resolves to its parent
- **WHEN** a child passage survives reranking and selection
- **THEN** the text supplied to generation is that child's parent text

#### Scenario: Matched child remains identifiable
- **WHEN** a parent is returned because one of its children matched
- **THEN** the matched child can be identified for display as the highlighted portion of the citation

### Requirement: Duplicate parent suppression

When multiple selected children share the same parent, the system SHALL include that parent only once and MUST backfill from remaining candidates to preserve the configured number of distinct contexts.

#### Scenario: Siblings collapse to one context
- **WHEN** several selected children belong to the same parent
- **THEN** that parent contributes exactly one context to generation

#### Scenario: Supply is backfilled after collapse
- **WHEN** duplicate suppression leaves fewer contexts than the configured minimum
- **THEN** the system draws further candidates from the reranked list until the minimum is satisfied or candidates are exhausted

### Requirement: Oversized parent handling

The system SHALL bound the size of context supplied to generation and MUST NOT exceed the configured context budget by passing an oversized parent verbatim.

#### Scenario: Oversized parent is reduced
- **WHEN** an expanded parent exceeds the configured size budget
- **THEN** the system supplies a bounded excerpt centred on the matched child instead of the full parent

### Requirement: Retrieval provenance

Every retrieved context SHALL carry provenance sufficient to cite it, and the system SHALL expose that provenance to callers alongside the retrieved text.

#### Scenario: Retrieved context is attributable
- **WHEN** retrieval returns contexts
- **THEN** each context is accompanied by its source document name and available location metadata

### Requirement: Retrieval observability

The system SHALL be able to report, for a single question, the intermediate results of each recall channel and of each ranking stage, so that retrieval quality can be inspected and compared.

#### Scenario: Per-stage inspection
- **WHEN** retrieval diagnostics are requested for a question
- **THEN** the lexical ranking, semantic ranking, fused ranking, and reranked ranking are each individually inspectable

#### Scenario: Channel contribution is determinable
- **WHEN** diagnostics are inspected
- **THEN** it is determinable which passages were contributed by which channel

### Requirement: Empty knowledge base

The system SHALL respond to a question against an empty knowledge base with an explicit "no content available" indication rather than an error or an ungrounded answer.

#### Scenario: Question with no ingested documents
- **WHEN** a question is asked and no documents are available
- **THEN** the system reports that no content is available for answering and does not generate an ungrounded answer