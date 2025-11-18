"""
Data preprocessing utilities for NLI and QA tasks.

This module contains functions for:
- Tokenizing and preprocessing NLI datasets
- Tokenizing and preprocessing QA datasets
- Computing evaluation metrics
- Custom trainer classes
"""

from .helpers import (
    prepare_dataset_nli,
    prepare_dataset_nli_hypothesis_only,
    prepare_train_dataset_qa,
    prepare_validation_dataset_qa,
    compute_accuracy,
    postprocess_qa_predictions,
    QuestionAnsweringTrainer,
)

__all__ = [
    'prepare_dataset_nli',
    'prepare_dataset_nli_hypothesis_only',
    'prepare_train_dataset_qa',
    'prepare_validation_dataset_qa',
    'compute_accuracy',
    'postprocess_qa_predictions',
    'QuestionAnsweringTrainer',
]

