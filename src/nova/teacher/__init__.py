"""Teacher package: teacher-model client and offline quality checks."""

from nova.teacher.client import TeacherClient, TeacherDisabled
from nova.teacher.quality import (
    generate_eval_questions,
    quality_report,
    review_candidate,
)

__all__ = [
    "TeacherClient",
    "TeacherDisabled",
    "generate_eval_questions",
    "quality_report",
    "review_candidate",
]
