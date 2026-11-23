# answer-generation

## Purpose

Defines how retrieved context is turned into a grounded, cited, incrementally delivered answer, and how follow-up questions in a conversation are made answerable on their own.

## Requirements

### Requirement: Grounded answer generation

The system SHALL answer questions using only the retrieved contexts supplied for that question, and SHALL be instructed to decline or qualify when the supplied contexts do not contain the answer.

#### Scenario: Answer is supported by supplied context
- **WHEN** supplied contexts contain the information needed to answer
- **THEN** the answer is consistent with those contexts

#### Scenario: Contexts do not contain the answer
- **WHEN** supplied contexts do not contain the information needed
- **THEN** the system indicates that the knowledge base does not cover the question rather than answering from unsupported background knowledge

### Requirement: Source attribution in answers

The answer presented to the user SHALL be accompanied by the sources actually supplied to the model, so the user can verify the basis of the answer.

#### Scenario: Sources accompany the answer
- **WHEN** an answer is produced from retrieved contexts
- **THEN** those contexts are presented to the user as citations alongside the answer

### Requirement: Streaming delivery

The system SHALL deliver the answer incrementally as it is generated, and MUST make retrieved sources available to the client before the answer text completes.

#### Scenario: Answer streams
- **WHEN** generation is in progress
- **THEN** partial answer text is delivered to the client as it becomes available

#### Scenario: Sources precede the completed answer
- **WHEN** the answer has been selected from retrieval
- **THEN** the sources are delivered to the client before the generation stream finishes

#### Scenario: Generation fails mid-stream
- **WHEN** the generation service fails after streaming has started
- **THEN** the client is informed that the answer was truncated and why, rather than the stream ending silently as if complete

### Requirement: Conversation context for follow-up questions

The system SHALL support multi-turn conversations in which a question depending on prior turns is reformulated into a self-contained query before retrieval.

#### Scenario: Dependent follow-up is reformulated
- **WHEN** a follow-up question refers to prior turns without stating its subject
- **THEN** the query used for retrieval is a self-contained restatement of that question

#### Scenario: Independent question is used as-is
- **WHEN** a question is already self-contained
- **THEN** the system uses it for retrieval without a reformulation step

#### Scenario: First turn requires no reformulation
- **WHEN** the conversation has no prior turns
- **THEN** no reformulation is performed

#### Scenario: Reformulation failure degrades safely
- **WHEN** reformulation fails or is unavailable
- **THEN** the original question is used for retrieval and the answer proceeds

### Requirement: Conversation history bounds

The system SHALL bound how much conversation history it retains for context, discarding the oldest turns when the bound is exceeded.

#### Scenario: History is capped
- **WHEN** a conversation exceeds the configured number of retained turns
- **THEN** the oldest turns are dropped from retrieval and generation context

### Requirement: No ungrounded answer for empty knowledge base

The system MUST NOT generate an answer when no retrievable content is available.

#### Scenario: Empty knowledge base
- **WHEN** a question is asked and no content is retrievable
- **THEN** the system returns a message indicating no content is available and performs no generation call

### Requirement: Generation failures are actionable

When the generation service is unreachable, misconfigured, or rejects the request, the system SHALL report the failure in terms of configuration or availability rather than surfacing an opaque transport error.

#### Scenario: Generation service unreachable
- **WHEN** the generation service cannot be reached
- **THEN** the reported error identifies the service and its configured endpoint

#### Scenario: Generation credentials are invalid
- **WHEN** the generation service rejects the configured credentials or model
- **THEN** the reported error identifies invalid credentials or an unknown model rather than an HTTP status alone
