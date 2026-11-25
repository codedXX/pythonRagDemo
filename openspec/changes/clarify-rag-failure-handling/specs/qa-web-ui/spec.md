## MODIFIED Requirements

### Requirement: Document inventory

The page SHALL display the documents currently available for questioning and SHALL let a user remove a document. The page SHALL distinguish a successfully loaded empty inventory from a failed inventory request or an invalid list response.

#### Scenario: Available documents are listed
- **WHEN** the page loads or after an ingestion completes
- **THEN** the currently available documents are displayed

#### Scenario: Removing a document
- **WHEN** a user removes a document from the list
- **THEN** the page reflects that it is no longer available

#### Scenario: Inventory is empty
- **WHEN** the inventory request succeeds with an empty documents array
- **THEN** the page displays the empty inventory message

#### Scenario: Inventory cannot be loaded
- **WHEN** the request fails, returns an unsuccessful HTTP status, or the successful response lacks a documents array
- **THEN** the document list displays a loading failure
