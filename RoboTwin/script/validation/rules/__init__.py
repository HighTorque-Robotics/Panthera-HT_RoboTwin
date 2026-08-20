"""Task-specific physics validation rules."""

from .move_pillbottle_pad import MovePillbottlePadRule
from .blocks_ranking_rgb import BlocksRankingRGBRule


RULES = {
    "blocks_ranking_rgb": BlocksRankingRGBRule,
    "move_pillbottle_pad": MovePillbottlePadRule,
}


def create_rule(task_name: str):
    try:
        return RULES[task_name]()
    except KeyError as exc:
        supported = ", ".join(sorted(RULES))
        raise ValueError(
            f"No physics validation rule for {task_name!r}; supported tasks: {supported}"
        ) from exc
