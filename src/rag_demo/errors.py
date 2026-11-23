"""Error types shared by the provider adapters.

The distinction that matters throughout this project is between a *transient*
failure (worth retrying) and a *configuration* failure (never worth retrying, and
the user must be told what to change). Collapsing both into "request failed"
would make a wrong model name look like a network blip.
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Base class for provider interaction failures."""


class ConfigurationIssue(ProviderError):
    """The provider rejected our credentials or model, or is not configured.

    Retrying cannot help: the configuration must change.
    """

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        self.message = message
        super().__init__(f"{provider}: {message}")


class TransientFailure(ProviderError):
    """A failure that may succeed if retried (rate limit, 5xx, timeout)."""

    def __init__(self, provider: str, message: str, status: int | None = None) -> None:
        self.provider = provider
        self.message = message
        self.status = status
        super().__init__(f"{provider}: {message}")


class ContractMismatch(ProviderError):
    """The provider answered, but not in the shape we depend on.

    This is reported rather than tolerated: silently guessing at an unexpected
    payload would produce wrong results far from the cause.
    """

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        self.message = message
        super().__init__(f"{provider}: {message}")


class Unavailable(ProviderError):
    """The provider could not be reached at all."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        self.message = message
        super().__init__(f"{provider}: {message}")