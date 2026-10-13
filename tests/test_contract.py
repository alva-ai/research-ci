import pytest

from research_ci import Contract, ContractError, validate_contract


def test_contract_is_only_a_validated_stage_boundary() -> None:
    validate_contract(
        Contract(
            mission="study a data relation",
            stages=("data", "feat"),
            scope=("two venues",),
            invariants=("point_in_time",),
        )
    )


def test_contract_rejects_out_of_order_stages() -> None:
    with pytest.raises(ContractError, match="order"):
        validate_contract(Contract(mission="study", stages=("feat", "data")))
