from typing import Any, Dict, Iterator, MutableMapping, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import torch
from pytorch_ie.annotations import Label
from pytorch_ie.core import AnnotationList, TaskEncoding, TaskModule, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.models.transformer_text_classification import (
    TransformerTextClassificationModelBatchOutput,
    TransformerTextClassificationModelStepBatchEncoding,
)
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias


class TextDocumentWithLabel(TextDocument):
    label: AnnotationList[Label] = annotation_field()


class TaskOutput(TypedDict, total=False):
    label: str
    probability: float


"""
workflow:
    document
        -> (input_encoding, target_encoding) -> task_encoding
            -> model_encoding -> model_output
        -> task_output
    -> document
"""


# Define input and output types
DocumentType: TypeAlias = TextDocumentWithLabel
InputEncodingType: TypeAlias = MutableMapping[str, Any]
TargetEncodingType: TypeAlias = int
ModelEncodingType: TypeAlias = TransformerTextClassificationModelStepBatchEncoding
ModelOutputType: TypeAlias = TransformerTextClassificationModelBatchOutput
TaskOutputType: TypeAlias = TaskOutput

TaskEncodingType: TypeAlias = TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType]


@TaskModule.register()
class SimpleTransformerTextClassificationTaskModule(
    TaskModule[
        DocumentType,
        InputEncodingType,
        TargetEncodingType,
        ModelEncodingType,
        ModelOutputType,
        TaskOutputType,
    ]
):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_to_id: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # Save all passed arguments. They will be available via self._config().
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # some tokenization and padding parameters
        self.truncation = truncation
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of

        # this will be prepared from the data or loaded from the config
        self.label_to_id = label_to_id or {}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def _config(self) -> Dict[str, Any]:
        """
        Add config entries. The config will be dumped when calling save_pretrained().
        Entries of the config will be passed to the constructor of this taskmodule when
        loading it with from_pretrained().
        """
        # add the label-to-id mapping to the config
        config = super()._config()
        config["label_to_id"] = self.label_to_id
        return config

    def prepare(self, documents: Sequence[DocumentType]) -> None:
        """
        Prepare the task module with training documents, e.g. collect all possible labels.
        """

        # create the label-to-id mapping
        labels = set()
        for document in documents:
            # all annotations of a document are hold in list like containers,
            # so we have to take its first element
            label_annotation = document.label[0]
            labels.add(label_annotation.label)

        # create the mapping, but spare the first index for the "O" (outside) class
        self.label_to_id = {label: i + 1 for i, label in enumerate(sorted(labels))}
        self.label_to_id["O"] = 0
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> TaskEncodingType:
        """
        Create one or multiple task encodings for the given document.
        """

        # tokenize the input text, this will be the input
        inputs = self.tokenizer(
            document.text,
            # we do not pad here, this will be done in collate()
            # when the actual batches are created
            padding=False,
            truncation=self.truncation,
            max_length=self.max_length,
        )

        return TaskEncoding(
            document=document,
            inputs=inputs,
        )

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> TargetEncodingType:
        """
        Create a target for a task encoding. This may use any annotations of the underlying document.
        """

        # as above, all annotations are hold in lists, so we have to take its first element
        label_annotation = task_encoding.document.label[0]
        # translate the textual label to the target id
        return self.label_to_id[label_annotation.label]

    def unbatch_output(
        self, model_output: TransformerTextClassificationModelBatchOutput
    ) -> Sequence[TaskOutputType]:
        """
        Convert one model output batch to a sequence of taskmodule outputs.
        """

        # get the logits from the model output
        logits = model_output["logits"]

        # convert the logits to "probabilities"
        probabilities = logits.softmax(dim=-1).detach().cpu().numpy()

        # get the max class index per example
        max_label_ids = np.argmax(probabilities, axis=-1)

        outputs = []
        for idx, label_id in enumerate(max_label_ids):
            # translate the label id back to the label text
            label = self.id_to_label[label_id]
            # get the probability and convert from tensor value to python float
            prob = float(probabilities[idx, label_id])
            # we create TransformerTextClassificationTaskOutput primarily for typing purposes,
            # a simple dict would also work
            result: TaskOutput = {
                "label": label,
                "probability": prob,
            }
            outputs.append(result)

        return outputs

    def create_annotations_from_output(
        self,
        task_encodings: TaskEncodingType,
        task_outputs: TaskOutputType,
    ) -> Iterator[Tuple[str, Label]]:
        """
        Convert a task output to annotations. The method has to yield tuples (annotation_name, annotation).
        """

        # just yield a single annotation (other tasks may need multiple annotations per task output)
        yield "label", Label(label=task_outputs["label"], score=task_outputs["probability"])

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> ModelEncodingType:
        """
        Convert a list of task encodings to a batch that will be passed to the model.
        """
        # get the inputs from the task encodings
        input_features = [task_encoding.inputs for task_encoding in task_encodings]

        # pad the inputs and return torch tensors
        inputs = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if task_encodings[0].has_targets:
            # convert the targets (label ids) to a tensor
            targets = torch.tensor(
                [task_encoding.targets for task_encoding in task_encodings], dtype=torch.int64
            )
        else:
            # during inference, we do not have any targets
            targets = None

        return inputs, targets
