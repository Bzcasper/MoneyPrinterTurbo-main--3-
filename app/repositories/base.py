"""
Base repository class with common CRUD operations
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_, desc, asc, func
from datetime import datetime
import logging

from app.models.database import Base

logger = logging.getLogger(__name__)

ModelType = TypeVar('ModelType', bound=Base)


class BaseRepository(Generic[ModelType], ABC):
    """Base repository with common CRUD operations"""
    
    def __init__(self, session: Session, model_class: type):
        self.session = session
        self.model_class = model_class
    
    def create(self, **kwargs) -> ModelType:
        """Create a new entity"""
        try:
            entity = self.model_class(**kwargs)
            self.session.add(entity)
            self.session.commit()
            self.session.refresh(entity)
            logger.debug(f"Created {self.model_class.__name__} with ID: {entity.id}")
            return entity
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error creating {self.model_class.__name__}: {e}")
            raise
    
    def get_by_id(self, entity_id: str) -> Optional[ModelType]:
        """Get entity by ID"""
        try:
            entity = self.session.query(self.model_class).filter(
                self.model_class.id == entity_id
            ).first()
            return entity
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model_class.__name__} by ID {entity_id}: {e}")
            return None
    
    def get_all(self, limit: int = None, offset: int = None) -> List[ModelType]:
        """Get all entities with optional pagination"""
        try:
            query = self.session.query(self.model_class)
            
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
                    
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all {self.model_class.__name__}: {e}")
            return []
    
    def update(self, entity_id: str, **kwargs) -> Optional[ModelType]:
        """Update entity by ID"""
        try:
            entity = self.get_by_id(entity_id)
            if not entity:
                return None
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            
            # Update timestamp if available
            if hasattr(entity, 'updated_at'):
                entity.updated_at = datetime.utcnow()
            
            self.session.commit()
            self.session.refresh(entity)
            logger.debug(f"Updated {self.model_class.__name__} with ID: {entity_id}")
            return entity
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error updating {self.model_class.__name__} {entity_id}: {e}")
            raise
    
    def delete(self, entity_id: str) -> bool:
        """Delete entity by ID"""
        try:
            entity = self.get_by_id(entity_id)
            if not entity:
                return False
            
            self.session.delete(entity)
            self.session.commit()
            logger.debug(f"Deleted {self.model_class.__name__} with ID: {entity_id}")
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error deleting {self.model_class.__name__} {entity_id}: {e}")
            return False
    
    def count(self, **filters) -> int:
        """Count entities with optional filters"""
        try:
            query = self.session.query(func.count(self.model_class.id))
            query = self._apply_filters(query, **filters)
            return query.scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error counting {self.model_class.__name__}: {e}")
            return 0
    
    def exists(self, entity_id: str) -> bool:
        """Check if entity exists"""
        try:
            return self.session.query(
                self.session.query(self.model_class).filter(
                    self.model_class.id == entity_id
                ).exists()
            ).scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error checking existence of {self.model_class.__name__} {entity_id}: {e}")
            return False
    
    def find_by(self, limit: int = None, offset: int = None, 
                order_by: str = None, order_dir: str = 'asc', **filters) -> List[ModelType]:
        """Find entities by filters"""
        try:
            query = self.session.query(self.model_class)
            
            # Apply filters
            query = self._apply_filters(query, **filters)
            
            # Apply ordering
            if order_by and hasattr(self.model_class, order_by):
                order_column = getattr(self.model_class, order_by)
                if order_dir.lower() == 'desc':
                    query = query.order_by(desc(order_column))
                else:
                    query = query.order_by(asc(order_column))
            
            # Apply pagination
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error finding {self.model_class.__name__}: {e}")
            return []
    
    def find_one_by(self, **filters) -> Optional[ModelType]:
        """Find single entity by filters"""
        results = self.find_by(limit=1, **filters)
        return results[0] if results else None
    
    def bulk_create(self, entities_data: List[Dict[str, Any]]) -> List[ModelType]:
        """Bulk create entities"""
        try:
            entities = []
            for data in entities_data:
                entity = self.model_class(**data)
                entities.append(entity)
                self.session.add(entity)
            
            self.session.commit()
            
            # Refresh all entities to get generated IDs
            for entity in entities:
                self.session.refresh(entity)
            
            logger.debug(f"Bulk created {len(entities)} {self.model_class.__name__} entities")
            return entities
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error bulk creating {self.model_class.__name__}: {e}")
            raise
    
    def bulk_update(self, updates: List[Dict[str, Any]]) -> bool:
        """Bulk update entities. Each update dict should contain 'id' and fields to update"""
        try:
            for update_data in updates:
                entity_id = update_data.pop('id')
                entity = self.get_by_id(entity_id)
                if entity:
                    for key, value in update_data.items():
                        if hasattr(entity, key):
                            setattr(entity, key, value)
                    
                    if hasattr(entity, 'updated_at'):
                        entity.updated_at = datetime.utcnow()
            
            self.session.commit()
            logger.debug(f"Bulk updated {len(updates)} {self.model_class.__name__} entities")
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error bulk updating {self.model_class.__name__}: {e}")
            return False
    
    def bulk_delete(self, entity_ids: List[str]) -> int:
        """Bulk delete entities by IDs"""
        try:
            deleted_count = self.session.query(self.model_class).filter(
                self.model_class.id.in_(entity_ids)
            ).delete(synchronize_session=False)
            
            self.session.commit()
            logger.debug(f"Bulk deleted {deleted_count} {self.model_class.__name__} entities")
            return deleted_count
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error bulk deleting {self.model_class.__name__}: {e}")
            return 0
    
    def _apply_filters(self, query: Query, **filters) -> Query:
        """Apply filters to query"""
        for field, value in filters.items():
            if hasattr(self.model_class, field):
                column = getattr(self.model_class, field)
                
                # Handle different filter types
                if isinstance(value, dict):
                    # Handle range filters like {'gte': 10, 'lte': 100}
                    if 'gte' in value:
                        query = query.filter(column >= value['gte'])
                    if 'lte' in value:
                        query = query.filter(column <= value['lte'])
                    if 'gt' in value:
                        query = query.filter(column > value['gt'])
                    if 'lt' in value:
                        query = query.filter(column < value['lt'])
                    if 'in' in value:
                        query = query.filter(column.in_(value['in']))
                    if 'like' in value:
                        query = query.filter(column.like(f"%{value['like']}%"))
                elif isinstance(value, list):
                    # Handle IN filters
                    query = query.filter(column.in_(value))
                else:
                    # Handle equality filters
                    query = query.filter(column == value)
        
        return query
    
    def paginate(self, page: int = 1, per_page: int = 20, **filters) -> Dict[str, Any]:
        """Paginate results with metadata"""
        try:
            # Calculate offset
            offset = (page - 1) * per_page
            
            # Get total count
            total = self.count(**filters)
            
            # Get items
            items = self.find_by(limit=per_page, offset=offset, **filters)
            
            # Calculate pagination metadata
            total_pages = (total + per_page - 1) // per_page
            has_next = page < total_pages
            has_prev = page > 1
            
            return {
                'items': items,
                'total': total,
                'page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'has_next': has_next,
                'has_prev': has_prev
            }
        except Exception as e:
            logger.error(f"Error paginating {self.model_class.__name__}: {e}")
            return {
                'items': [],
                'total': 0,
                'page': page,
                'per_page': per_page,
                'total_pages': 0,
                'has_next': False,
                'has_prev': False
            }
    
    def search(self, search_term: str, search_fields: List[str], 
               limit: int = None, **filters) -> List[ModelType]:
        """Search entities across specified fields"""
        try:
            query = self.session.query(self.model_class)
            
            # Apply base filters
            query = self._apply_filters(query, **filters)
            
            # Build search conditions
            search_conditions = []
            for field in search_fields:
                if hasattr(self.model_class, field):
                    column = getattr(self.model_class, field)
                    search_conditions.append(column.like(f"%{search_term}%"))
            
            if search_conditions:
                query = query.filter(or_(*search_conditions))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error searching {self.model_class.__name__}: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics for the entity"""
        try:
            total_count = self.count()
            
            stats = {
                'total_count': total_count
            }
            
            # Add timestamp-based stats if available
            if hasattr(self.model_class, 'created_at'):
                # Count created today
                today = datetime.utcnow().date()
                created_today = self.count(
                    created_at={'gte': datetime.combine(today, datetime.min.time())}
                )
                stats['created_today'] = created_today
                
                # Count created this week
                from datetime import timedelta
                week_ago = datetime.utcnow() - timedelta(days=7)
                created_this_week = self.count(created_at={'gte': week_ago})
                stats['created_this_week'] = created_this_week
            
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics for {self.model_class.__name__}: {e}")
            return {'total_count': 0}