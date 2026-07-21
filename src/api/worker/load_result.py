from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoadResult:
    loaded: bool
    problems: list[str] = field(default_factory=list)
