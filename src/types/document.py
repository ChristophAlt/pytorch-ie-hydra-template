import dataclasses
from typing import Any, Dict, Optional, Tuple

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, Document, annotation_field

from src.types.annotation import Attribution


@dataclasses.dataclass
class _Metadata:
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TokenBasedDocument(Document):
    tokens: Tuple[str, ...]


@dataclasses.dataclass
class TextBasedDocument(Document):
    text: str


@dataclasses.dataclass
class TokenDocumentWithEntitiesAndRelations(_Metadata, TokenBasedDocument):
    entities: AnnotationList[Span] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TokenDocumentWithLabeledEntitiesAndRelations(_Metadata, TokenBasedDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TextDocumentWithEntityMentions(_Metadata, TextBasedDocument):
    entity_mentions: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithEntitiesAndRelations(_Metadata, TextBasedDocument):
    """Possible input class for TransformerRETextClassificationTaskModule."""

    entities: AnnotationList[Span] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesAndRelations(_Metadata, TextBasedDocument):
    """Possible input class for TransformerRETextClassificationTaskModule."""

    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class DocumentWithEntitiesRelationsAndLabeledPartitions(_Metadata, TextBasedDocument):
    """Possible input class for TransformerRETextClassificationTaskModule."""

    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
    partitions: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class BratDocument(_Metadata, TextBasedDocument):
    """Possible input class for TransformerRETextClassificationTaskModule."""

    spans: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="spans")
    span_attributions: AnnotationList[Attribution] = annotation_field(target="spans")
    relation_attributions: AnnotationList[Attribution] = annotation_field(target="relations")