"""Contract-level errors."""


class ContractError(ValueError):
    """Raised when a payload does not satisfy a versioned contract."""
