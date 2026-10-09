from research_ci.workflow.contract import ContractCommands
from research_ci.workflow.data import DataCommands
from research_ci.workflow.feat import FeatCommands


class Idea(ContractCommands, DataCommands, FeatCommands):
    """One idea's lifecycle: the shared machinery in ``IdeaBase`` and each stage's commands from its own module."""
