from dataclasses import dataclass

from research_ci.workflow.machine import Kind, WorkflowError

CONTRACT = "contract"
LAYERS = ("data", "feat", "signal", "algo")
CHECKPOINTS = ("seal", "forward", "capital", "monitor")


@dataclass(frozen=True)
class Template:
    name: str
    layers: tuple[str, ...]
    optional: frozenset[str] = frozenset()

    @property
    def stages(self) -> tuple[str, ...]:
        return (CONTRACT, *self.layers, *CHECKPOINTS)


TEMPLATES = {
    t.name: t
    for t in (
        Template("factor", ("data", "feat", "signal", "algo"), optional=frozenset({"signal", "algo"})),
        Template("rule", ("data", "signal", "algo")),
        Template("event", ("data", "feat", "signal", "algo")),
        Template("taker_arb", ("data", "signal", "algo"), optional=frozenset({"signal"})),
        Template("maker_arb", ("data", "feat", "signal", "algo")),
    )
}


def get_template(name: str) -> Template:
    if name not in TEMPLATES:
        raise WorkflowError(f"unknown template {name!r}; choose one of {', '.join(TEMPLATES)}")
    return TEMPLATES[name]


def kind_of(stage: str) -> Kind:
    if stage == CONTRACT:
        return Kind.CONTRACT
    if stage in LAYERS:
        return Kind.LAYER
    if stage in CHECKPOINTS:
        return Kind.CHECKPOINT
    raise WorkflowError(f"unknown stage {stage!r}")
