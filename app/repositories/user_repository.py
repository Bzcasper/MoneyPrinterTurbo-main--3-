"""
User repository for user management and session handling
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc
from datetime import datetime, timedelta
import logging
import secrets

from app.models.database import User, UserSession
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class UserRepository(BaseRepository[User]):
    """Repository for user operations"""
    
    def __init__(self, session: Session):
        super().__init__(session, User)
    
    def create_user(self, username: str, email: str = None, 
                   preferences: Dict[str, Any] = None) -> User:
        """Create a new user"""
        try:
            user = User(
                username=username,
                email=email,
                preferences=preferences or {}
            )
            
            self.session.add(user)
            self.session.commit()
            self.session.refresh(user)
            
            logger.info(f"Created user '{username}'")
            return user
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating user: {e}")
            raise
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            return self.session.query(User).filter(
                User.username == username
            ).first()
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            return self.session.query(User).filter(
                User.email == email
            ).first()
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    def update_user_preferences(self, user_id: str, 
                              preferences: Dict[str, Any]) -> Optional[User]:
        """Update user preferences"""
        try:
            user = self.get_by_id(user_id)
            if not user:
                return None
            
            # Merge with existing preferences
            current_prefs = user.preferences or {}
            current_prefs.update(preferences)
            user.preferences = current_prefs
            user.updated_at = datetime.utcnow()
            
            self.session.commit()
            self.session.refresh(user)
            
            logger.debug(f"Updated preferences for user {user_id}")
            return user
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating user preferences: {e}")
            raise
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account"""
        try:
            user = self.update(user_id, is_active=False)
            if user:
                logger.info(f"Deactivated user {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deactivating user: {e}")
            return False
    
    def activate_user(self, user_id: str) -> bool:
        """Activate a user account"""
        try:
            user = self.update(user_id, is_active=True)
            if user:
                logger.info(f"Activated user {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error activating user: {e}")
            return False
    
    def get_active_users(self, limit: int = None) -> List[User]:
        """Get active users"""
        try:
            query = self.session.query(User).filter(User.is_active == True)
            
            if limit:
                query = query.limit(limit)
            
            return query.order_by(User.username).all()
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            return []
    
    def search_users(self, search_term: str, active_only: bool = True,
                    limit: int = 50) -> List[User]:
        """Search users by username or email"""
        try:
            query = self.session.query(User)
            
            # Text search
            search_filter = or_(
                User.username.like(f"%{search_term}%"),
                User.email.like(f"%{search_term}%")
            )
            query = query.filter(search_filter)
            
            # Active filter
            if active_only:
                query = query.filter(User.is_active == True)
            
            query = query.order_by(User.username)
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error searching users: {e}")
            return []


class SessionRepository(BaseRepository[UserSession]):
    """Repository for user session operations"""
    
    def __init__(self, session: Session):
        super().__init__(session, UserSession)
    
    def create_session(self, user_id: str = None, ip_address: str = None,
                      user_agent: str = None, expires_in_hours: int = 24,
                      session_data: Dict[str, Any] = None) -> UserSession:
        """Create a new user session"""
        try:
            # Generate secure session token
            session_token = secrets.token_urlsafe(32)
            
            # Calculate expiration
            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
            
            session = UserSession(
                user_id=user_id,
                session_token=session_token,
                ip_address=ip_address,
                user_agent=user_agent,
                expires_at=expires_at,
                session_data=session_data or {}
            )
            
            self.session.add(session)
            self.session.commit()
            self.session.refresh(session)
            
            logger.debug(f"Created session for user {user_id}")
            return session
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating session: {e}")
            raise
    
    def get_session_by_token(self, session_token: str) -> Optional[UserSession]:
        """Get session by token"""
        try:
            return self.session.query(UserSession).filter(
                and_(
                    UserSession.session_token == session_token,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow()
                )
            ).first()
        except Exception as e:
            logger.error(f"Error getting session by token: {e}")
            return None
    
    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[UserSession]:
        """Get all sessions for a user"""
        try:
            query = self.session.query(UserSession).filter(
                UserSession.user_id == user_id
            )
            
            if active_only:
                query = query.filter(
                    and_(
                        UserSession.is_active == True,
                        UserSession.expires_at > datetime.utcnow()
                    )
                )
            
            return query.order_by(desc(UserSession.created_at)).all()
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    def update_session_activity(self, session_token: str,
                               session_data: Dict[str, Any] = None) -> Optional[UserSession]:
        """Update session last activity and data"""
        try:
            session = self.get_session_by_token(session_token)
            if not session:
                return None
            
            session.last_activity = datetime.utcnow()
            
            if session_data:
                current_data = session.session_data or {}
                current_data.update(session_data)
                session.session_data = current_data
            
            self.session.commit()
            self.session.refresh(session)
            
            return session
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating session activity: {e}")
            raise
    
    def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a session"""
        try:
            session = self.session.query(UserSession).filter(
                UserSession.session_token == session_token
            ).first()
            
            if session:
                session.is_active = False
                self.session.commit()
                logger.debug(f"Invalidated session {session_token}")
                return True
            
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error invalidating session: {e}")
            return False
    
    def invalidate_user_sessions(self, user_id: str, 
                                exclude_token: str = None) -> int:
        """Invalidate all sessions for a user (except optionally one)"""
        try:
            query = self.session.query(UserSession).filter(
                and_(
                    UserSession.user_id == user_id,
                    UserSession.is_active == True
                )
            )
            
            if exclude_token:
                query = query.filter(UserSession.session_token != exclude_token)
            
            count = query.update({'is_active': False}, synchronize_session=False)
            self.session.commit()
            
            logger.info(f"Invalidated {count} sessions for user {user_id}")
            return count
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error invalidating user sessions: {e}")
            return 0
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        try:
            count = self.session.query(UserSession).filter(
                UserSession.expires_at <= datetime.utcnow()
            ).delete(synchronize_session=False)
            
            self.session.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired sessions")
            
            return count
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    def get_session_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get session usage statistics"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Total sessions in period
            total_sessions = self.session.query(UserSession).filter(
                UserSession.created_at >= start_date
            ).count()
            
            # Active sessions
            active_sessions = self.session.query(UserSession).filter(
                and_(
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow()
                )
            ).count()
            
            # Sessions by day
            daily_sessions = self.session.query(
                func.date(UserSession.created_at).label('date'),
                func.count(UserSession.id).label('count')
            ).filter(
                UserSession.created_at >= start_date
            ).group_by(func.date(UserSession.created_at)).all()
            
            # Average session duration (for completed sessions)
            avg_duration = self.session.query(
                func.avg(
                    func.julianday(UserSession.last_activity) - 
                    func.julianday(UserSession.created_at)
                ) * 24 * 60  # Convert to minutes
            ).filter(
                and_(
                    UserSession.created_at >= start_date,
                    UserSession.last_activity.isnot(None)
                )
            ).scalar()
            
            # Unique users with sessions
            unique_users = self.session.query(
                func.count(func.distinct(UserSession.user_id))
            ).filter(
                and_(
                    UserSession.created_at >= start_date,
                    UserSession.user_id.isnot(None)
                )
            ).scalar()
            
            return {
                'period_days': days,
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'unique_users': unique_users or 0,
                'avg_session_duration_minutes': round(avg_duration or 0, 2),
                'daily_sessions': [
                    {
                        'date': str(day.date),
                        'session_count': day.count
                    } for day in daily_sessions
                ]
            }
        except Exception as e:
            logger.error(f"Error getting session statistics: {e}")
            return {}
    
    def extend_session(self, session_token: str, 
                      extend_hours: int = 24) -> Optional[UserSession]:
        """Extend session expiration"""
        try:
            session = self.get_session_by_token(session_token)
            if not session:
                return None
            
            session.expires_at = datetime.utcnow() + timedelta(hours=extend_hours)
            session.last_activity = datetime.utcnow()
            
            self.session.commit()
            self.session.refresh(session)
            
            logger.debug(f"Extended session {session_token} by {extend_hours} hours")
            return session
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error extending session: {e}")
            raise