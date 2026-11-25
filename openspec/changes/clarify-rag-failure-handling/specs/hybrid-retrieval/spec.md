## MODIFIED Requirements

### Requirement: Dual recall

The system SHALL retrieve candidate passages using both a lexical (keyword-based) channel and a semantic (embedding-based) channel, and MUST issue both recalls for every question. A lexical execution failure SHALL end the current question with a retrieval error.

#### Scenario: Both channels contribute
- **WHEN** a question is submitted for retrieval
- **THEN** the system produces a lexical candidate set and a semantic candidate set before any fusion

#### Scenario: One channel returns nothing
- **WHEN** either channel returns no candidates
- **THEN** the system continues with the remaining channel's results rather than failing the request

#### Scenario: Lexical engine raises during search
- **WHEN** the lexical search fails during execution, including when semantic candidates are available
- **THEN** the question stream reports a retrieval error and ends before answer generation
