# schema_intelligence/repository.py
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ColumnRoleDetection, DatasetRelationship, DuplicateColumnPair


class RelationshipRepository:
    def __init__(self, db: Session):
        self.db = db

    def add_relationships(self, relationships: list[DatasetRelationship]) -> list[DatasetRelationship]:
        if not relationships:
            return []
        self.db.add_all(relationships)
        self.db.commit()
        for r in relationships:
            self.db.refresh(r)
        return relationships

    def list_for_dataset(self, dataset_id: str) -> list[DatasetRelationship]:
        stmt = select(DatasetRelationship).where(
            (DatasetRelationship.source_dataset_id == dataset_id)
            | (DatasetRelationship.target_dataset_id == dataset_id)
        )
        return list(self.db.execute(stmt).scalars().all())


class ColumnRoleDetectionRepository:
    def __init__(self, db: Session):
        self.db = db

    def replace_for_dataset(self, dataset_id: str, detections: list[ColumnRoleDetection]) -> list[ColumnRoleDetection]:
        """Re-analyzing a dataset should REPLACE its prior detections, not
        pile up duplicates on every re-run — so this deletes the dataset's
        existing rows before inserting the new set, in one transaction."""
        self.db.query(ColumnRoleDetection).filter(ColumnRoleDetection.dataset_id == dataset_id).delete()
        if detections:
            self.db.add_all(detections)
        self.db.commit()
        for d in detections:
            self.db.refresh(d)
        return detections

    def list_for_dataset(self, dataset_id: str) -> list[ColumnRoleDetection]:
        stmt = select(ColumnRoleDetection).where(ColumnRoleDetection.dataset_id == dataset_id)
        return list(self.db.execute(stmt).scalars().all())

    def list_for_column(self, dataset_id: str, column_name: str) -> list[ColumnRoleDetection]:
        stmt = select(ColumnRoleDetection).where(
            ColumnRoleDetection.dataset_id == dataset_id,
            ColumnRoleDetection.column_name == column_name,
        ).order_by(ColumnRoleDetection.confidence.desc())
        return list(self.db.execute(stmt).scalars().all())


class DuplicateColumnRepository:
    def __init__(self, db: Session):
        self.db = db

    def replace_for_dataset(self, dataset_id: str, pairs: list[DuplicateColumnPair]) -> list[DuplicateColumnPair]:
        self.db.query(DuplicateColumnPair).filter(DuplicateColumnPair.dataset_id == dataset_id).delete()
        if pairs:
            self.db.add_all(pairs)
        self.db.commit()
        for p in pairs:
            self.db.refresh(p)
        return pairs

    def list_for_dataset(self, dataset_id: str) -> list[DuplicateColumnPair]:
        stmt = select(DuplicateColumnPair).where(DuplicateColumnPair.dataset_id == dataset_id)
        return list(self.db.execute(stmt).scalars().all())
