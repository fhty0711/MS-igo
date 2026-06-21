"""Backend registry."""

from .generic_scenario import GenericScenarioBackend


BACKENDS = {
    "generic_scenario": GenericScenarioBackend,
}


def get_backend(name):
    try:
        return BACKENDS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown backend {name!r}. Available: {available}") from exc
