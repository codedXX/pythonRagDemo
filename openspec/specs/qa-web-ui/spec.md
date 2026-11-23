# qa-web-ui

## Purpose

Defines the browser-facing experience: a self-contained page for asking questions, uploading documents, watching ingestion progress, and inspecting the sources behind each answer.

## Requirements

### Requirement: Page is served as static assets

The question-answering interface SHALL be served as static files by the same service that exposes the API, and MUST NOT require a separate build step or a separate front-end server to run.

#### Scenario: Page loads from the running service
- **WHEN** a user opens the service's root URL
- **THEN** the interface loads without any build or bundling step

#### Scenario: Secrets never reach the browser
- **WHEN** the page communicates with backend capabilities
- **THEN** no provider credentials, API keys, or upstream service endpoints are present in anything delivered to the browser

### Requirement: Question answering interface

The page SHALL let a user submit a question, view the answer as it streams, and keep the exchange in a conversation view that supports follow-up questions.

#### Scenario: Answer appears incrementally
- **WHEN** a user submits a question
- **THEN** the answer text becomes visible progressively rather than appearing only once complete

#### Scenario: Sources are shown per answer
- **WHEN** an answer is displayed
- **THEN** the sources used for that answer are shown with it, identifying at least the source document and available location metadata

#### Scenario: Follow-up question
- **WHEN** a user submits a further question in the same conversation
- **THEN** the earlier exchanges remain visible and the new question is answered in their light

#### Scenario: Empty knowledge base is explained
- **WHEN** the user asks a question with no documents ingested
- **THEN** the page explains that no content is available instead of showing a failed or fabricated answer

### Requirement: Document upload

The page SHALL let a user select or drag documents to be ingested, SHALL accept PDF and Markdown, and SHALL report rejection of unsupported or unusable files with a clear message.

#### Scenario: Supported file is accepted
- **WHEN** a user selects or drops a PDF or Markdown file
- **THEN** ingestion starts for that file

#### Scenario: Unsupported file is reported
- **WHEN** a user selects a file of an unsupported type
- **THEN** the page reports that the file type is unsupported and lists the supported types

#### Scenario: Unusable document is reported
- **WHEN** a document cannot be processed, for example a scanned PDF
- **THEN** the page reports the specific reason rather than a generic failure

### Requirement: Ingestion progress feedback

The page SHALL display ingestion progress for each file, distinguishing active progress from completion and failure.

#### Scenario: Progress is visible during processing
- **WHEN** a file is being ingested
- **THEN** the page shows which stage is active and that work is ongoing

#### Scenario: Completion is distinguishable
- **WHEN** ingestion finishes successfully
- **THEN** the page shows that the file is ready to be queried

#### Scenario: Failure is distinguishable
- **WHEN** ingestion fails
- **THEN** the page marks the file as failed and shows the failure reason

#### Scenario: Multiple files in flight
- **WHEN** several files are uploaded together
- **THEN** each file reports its own progress independently

### Requirement: Document inventory

The page SHALL display the documents currently available for questioning and SHALL let a user remove a document.

#### Scenario: Available documents are listed
- **WHEN** the page loads or after an ingestion completes
- **THEN** the currently available documents are displayed

#### Scenario: Removing a document
- **WHEN** a user removes a document from the list
- **THEN** the page reflects that it is no longer available

### Requirement: Service health visibility

The page SHALL surface the state of the system's external dependencies so that a user can distinguish a configuration problem from a genuine failure.

#### Scenario: Dependency problem is surfaced
- **WHEN** an external dependency such as the embedding, reranking, or generation service is misconfigured or unreachable
- **THEN** the page indicates which dependency is affected before the user attempts a question or upload

### Requirement: Usable on a typical desktop browser

The interface SHALL function in a current desktop browser without installing extensions, and SHALL remain usable when the conversation and document lists grow.

#### Scenario: Long conversation remains usable
- **WHEN** a conversation contains many exchanges
- **THEN** the conversation area remains navigable

#### Scenario: Long answers render correctly
- **WHEN** an answer contains code blocks, lists, or tables
- **THEN** the answer is rendered legibly rather than as raw markup
