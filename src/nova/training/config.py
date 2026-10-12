"""Training run configuration.

Pure dataclass + validation; importing this module never requires torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class TrainingConfig:
    """Hyperparameters and run settings for a training job."""

    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    sequence_length: int = 2048
    learning_rate: float = 2e-5
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    seed: int = 42

    def validate(self) -> "TrainingConfig":
        """Raise ValueError if any setting is out of range."""
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) \
                or self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive int, got {self.batch_size!r}"
            )
        if not isinstance(self.gradient_accumulation_steps, int) \
                or isinstance(self.gradient_accumulation_steps, bool) \
                or self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be a positive int, "
                f"got {self.gradient_accumulation_steps!r}"
            )
        if not isinstance(self.sequence_length, int) \
                or isinstance(self.sequence_length, bool) \
                or self.sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be a positive int, got {self.sequence_length!r}"
            )
        if not isinstance(self.learning_rate, (int, float)) \
                or self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be a positive number, got {self.learning_rate!r}"
            )
        if not isinstance(self.checkpoint_dir, str) or not self.checkpoint_dir.strip():
            raise ValueError(
                f"checkpoint_dir must be a non-empty string, got {self.checkpoint_dir!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        return self

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        """Build from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in dict(data).items() if k in known}
        return cls(**kwargs)

    @classmethod
    def from_config(cls, section: dict) -> "TrainingConfig":
        """Build from a config section dict (e.g. ``config.section("training")``)."""
        return cls.from_dict(section)

    def to_dict(self) -> dict:
        """Plain-dict representation (JSON serializable)."""
        return asdict(self)
