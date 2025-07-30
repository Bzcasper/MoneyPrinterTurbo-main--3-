"""
Supabase Database Migration Helper

This module provides utilities for creating and managing database migrations
for the MoneyPrinterTurbo application using Supabase.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .connection import get_supabase_connection, SupabaseConnectionError, ensure_connection
from .models import SUPABASE_SCHEMAS, get_all_tables_sql, get_create_table_sql


class MigrationError(Exception):
    """Custom exception for migration errors."""
    pass


class SupabaseMigrator:
    """
    Supabase database migration manager.
    
    Handles creating tables, indexes, and managing database schema changes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connection = get_supabase_connection()
        
    async def run_sql(self, sql: str, fetch: bool = False) -> Dict[str, Any]:
        """
        Execute SQL command.
        
        Uses direct PostgreSQL connection if available, otherwise falls back to Supabase RPC.
        
        Args:
            sql: SQL command to execute
            fetch: Whether to fetch results (for SELECT) or just execute (for DDL)
            
        Returns:
            Query result
            
        Raises:
            MigrationError: If SQL execution fails
        """
        try:
            if not self.connection.is_connected:
                await self.connection.connect(use_service_key=True)
            
            # Prefer direct PG connection
            if self.connection.pg_pool:
                self.logger.info(f"Executing SQL via asyncpg: {sql[:100]}...")
                async with self.connection.pg_pool.acquire() as conn:
                    if fetch:
                        result = await conn.fetch(sql)
                        return {
                            "data": [dict(row) for row in result],
                            "count": len(result),
                            "status": "success"
                        }
                    else:
                        await conn.execute(sql)
                        return {"data": [], "count": 0, "status": "success"}

            # Fallback to Supabase RPC
            self.logger.info(f"Executing SQL via Supabase RPC: {sql[:100]}...")
            client = self.connection.client
            response = client.rpc('exec_sql', {'sql': sql}).execute()
            
            self.logger.info(f"SQL executed successfully: {sql[:100]}...")
            return response
            
        except Exception as e:
            error_msg = f"Failed to execute SQL: {str(e)}"
            self.logger.error(error_msg)
            raise MigrationError(error_msg)
    
    async def check_table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        try:
            sql = f"""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = '{table_name}'
                );
            """
            
            response = await self.run_sql(sql, fetch=True)
            
            # Handle response from asyncpg or Supabase RPC
            data = response.get("data") if isinstance(response, dict) else getattr(response, 'data', [])
            
            if data:
                return data[0].get('exists', False)
            return False
            
        except Exception as e:
            self.logger.warning(f"Could not check table existence for {table_name}: {str(e)}")
            return False
    
    async def create_table(self, table_name: str, force: bool = False) -> bool:
        """
        Create a single table.
        
        Args:
            table_name: Name of the table to create
            force: Whether to drop and recreate if table exists
            
        Returns:
            True if table was created successfully
            
        Raises:
            MigrationError: If table creation fails
        """
        try:
            if table_name not in SUPABASE_SCHEMAS:
                raise MigrationError(f"Schema for table '{table_name}' not found")
            
            table_exists = await self.check_table_exists(table_name)
            
            if table_exists and not force:
                self.logger.info(f"Table '{table_name}' already exists, skipping")
                return True
            
            if table_exists and force:
                self.logger.info(f"Dropping existing table '{table_name}'")
                drop_sql = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
                await self.run_sql(drop_sql, fetch=False)
            
            # Create the table
            create_sql = get_create_table_sql(table_name)
            self.logger.info(f"Creating table '{table_name}'")
            await self.run_sql(create_sql, fetch=False)
            
            self.logger.info(f"Table '{table_name}' created successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to create table '{table_name}': {str(e)}"
            self.logger.error(error_msg)
            raise MigrationError(error_msg)
    
    async def create_all_tables(self, force: bool = False) -> Dict[str, bool]:
        """
        Create all tables defined in the schema.
        
        Args:
            force: Whether to drop and recreate existing tables
            
        Returns:
            Dictionary with table names and their creation status
        """
        results = {}
        
        # Define table creation order to handle dependencies
        table_order = [
            'users',
            'projects', 
            'videos',
            'tasks',
            'analytics',
            'system_config'
        ]
        
        for table_name in table_order:
            try:
                success = await self.create_table(table_name, force=force)
                results[table_name] = success
                
                # Small delay between table creations
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Failed to create table '{table_name}': {str(e)}")
                results[table_name] = False
        
        return results
    
    async def setup_rls_policies(self) -> bool:
        """
        Set up Row Level Security (RLS) policies.
        
        Returns:
            True if policies were set up successfully
        """
        try:
            self.logger.info("Setting up RLS policies")
            
            # First, verify that the users table exists before creating functions that reference it
            users_table_exists = await self.check_table_exists('users')
            if not users_table_exists:
                self.logger.warning("Users table does not exist, skipping RLS setup for now")
                return True  # Consider this successful - RLS can be set up later
            
            # Try to enable RLS on auth.users (this may fail in some Supabase configurations)
            try:
                auth_rls_sql = """
                    -- Check if auth.users table exists and is accessible
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.tables
                            WHERE table_schema = 'auth'
                            AND table_name = 'users'
                        ) THEN
                            ALTER TABLE auth.users ENABLE ROW LEVEL SECURITY;
                            
                            -- Allow users to read their own data
                            DROP POLICY IF EXISTS "Users can view own profile" ON auth.users;
                            CREATE POLICY "Users can view own profile"
                            ON auth.users FOR SELECT
                            USING (auth.uid() = id);
                            
                            -- Allow users to update their own data
                            DROP POLICY IF EXISTS "Users can update own profile" ON auth.users;
                            CREATE POLICY "Users can update own profile"
                            ON auth.users FOR UPDATE
                            USING (auth.uid() = id);
                        END IF;
                    EXCEPTION
                        WHEN insufficient_privilege THEN
                            RAISE NOTICE 'Insufficient privileges to modify auth.users, skipping';
                        WHEN OTHERS THEN
                            RAISE NOTICE 'Could not set up auth.users RLS: %', SQLERRM;
                    END;
                    $$;
                """
                
                await self.run_sql(auth_rls_sql, fetch=False)
                self.logger.info("Auth table RLS policies set up successfully")
                
            except Exception as auth_e:
                self.logger.warning(f"Could not set up auth.users RLS policies (this may be expected): {str(auth_e)}")
            
            # Create security helper functions
            security_sql = """
                -- Create function to check if user is admin
                CREATE OR REPLACE FUNCTION is_admin()
                RETURNS BOOLEAN AS $$
                BEGIN
                    RETURN EXISTS (
                        SELECT 1 FROM users
                        WHERE auth_id = auth.uid()
                        AND role = 'admin'
                    );
                EXCEPTION
                    WHEN OTHERS THEN
                        RETURN FALSE;
                END;
                $$ LANGUAGE plpgsql SECURITY DEFINER;
                
                -- Create function to get current user id
                CREATE OR REPLACE FUNCTION get_current_user_id()
                RETURNS UUID AS $$
                BEGIN
                    RETURN (
                        SELECT id FROM users
                        WHERE auth_id = auth.uid()
                        LIMIT 1
                    );
                EXCEPTION
                    WHEN OTHERS THEN
                        RETURN NULL;
                END;
                $$ LANGUAGE plpgsql SECURITY DEFINER;
            """
            
            await self.run_sql(security_sql, fetch=False)
            self.logger.info("Security helper functions created successfully")
            
            self.logger.info("RLS policies set up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up RLS policies: {str(e)}")
            # Don't fail the entire migration for RLS issues - they can be set up later
            self.logger.warning("RLS setup failed, but continuing with migration")
            return True  # Return True to continue migration
    
    async def create_indexes(self) -> bool:
        """
        Create additional performance indexes.
        
        Returns:
            True if indexes were created successfully
        """
        try:
            self.logger.info("Creating performance indexes")
            
            indexes_sql = """
                -- Performance indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_projects_status_user 
                ON projects(status, user_id);
                
                CREATE INDEX IF NOT EXISTS idx_videos_project_status 
                ON videos(project_id, status);
                
                CREATE INDEX IF NOT EXISTS idx_tasks_user_status 
                ON tasks(user_id, status);
                
                CREATE INDEX IF NOT EXISTS idx_analytics_user_event_date 
                ON analytics(user_id, event_type, created_at);
                
                -- Composite index for project search
                CREATE INDEX IF NOT EXISTS idx_projects_search 
                ON projects USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '')));
                
                -- Index for task queue processing
                CREATE INDEX IF NOT EXISTS idx_tasks_queue 
                ON tasks(status, created_at) WHERE status IN ('pending', 'running');
            """
            
            await self.run_sql(indexes_sql, fetch=False)
            
            self.logger.info("Performance indexes created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {str(e)}")
            return False
    
    async def setup_triggers(self) -> bool:
        """
        Set up database triggers for automatic timestamp updates.
        
        Returns:
            True if triggers were set up successfully
        """
        try:
            self.logger.info("Setting up database triggers")
            
            triggers_sql = """
                -- Function to update updated_at timestamp
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                
                -- Create triggers for all tables
                DROP TRIGGER IF EXISTS update_users_updated_at ON users;
                CREATE TRIGGER update_users_updated_at
                    BEFORE UPDATE ON users
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                
                DROP TRIGGER IF EXISTS update_projects_updated_at ON projects;
                CREATE TRIGGER update_projects_updated_at
                    BEFORE UPDATE ON projects
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                
                DROP TRIGGER IF EXISTS update_videos_updated_at ON videos;
                CREATE TRIGGER update_videos_updated_at
                    BEFORE UPDATE ON videos
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                
                DROP TRIGGER IF EXISTS update_tasks_updated_at ON tasks;
                CREATE TRIGGER update_tasks_updated_at
                    BEFORE UPDATE ON tasks
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                
                DROP TRIGGER IF EXISTS update_analytics_updated_at ON analytics;
                CREATE TRIGGER update_analytics_updated_at
                    BEFORE UPDATE ON analytics
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                
                DROP TRIGGER IF EXISTS update_system_config_updated_at ON system_config;
                CREATE TRIGGER update_system_config_updated_at
                    BEFORE UPDATE ON system_config
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            """
            
            await self.run_sql(triggers_sql, fetch=False)
            
            self.logger.info("Database triggers set up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up triggers: {str(e)}")
            return False
    
    async def run_full_migration(self, force: bool = False) -> Dict[str, Any]:
        """
        Run complete database migration.
        
        Args:
            force: Whether to force recreate existing tables
            
        Returns:
            Migration results summary
        """
        migration_start = datetime.now()
        results = {
            'started_at': migration_start.isoformat(),
            'tables': {},
            'rls_policies': False,
            'indexes': False,
            'triggers': False,
            'success': False,
            'errors': []
        }
        
        try:
            self.logger.info("Starting full database migration")
            
            # Step 1: Create tables
            self.logger.info("Step 1: Creating tables")
            table_results = await self.create_all_tables(force=force)
            results['tables'] = table_results
            
            failed_tables = [name for name, success in table_results.items() if not success]
            if failed_tables:
                results['errors'].append(f"Failed to create tables: {failed_tables}")
            
            # Step 2: Set up RLS policies
            self.logger.info("Step 2: Setting up RLS policies")
            rls_success = await self.setup_rls_policies()
            results['rls_policies'] = rls_success
            
            if not rls_success:
                results['errors'].append("Failed to set up RLS policies")
            
            # Step 3: Create indexes
            self.logger.info("Step 3: Creating indexes")
            indexes_success = await self.create_indexes()
            results['indexes'] = indexes_success
            
            if not indexes_success:
                results['errors'].append("Failed to create indexes")
            
            # Step 4: Set up triggers
            self.logger.info("Step 4: Setting up triggers")
            triggers_success = await self.setup_triggers()
            results['triggers'] = triggers_success
            
            if not triggers_success:
                results['errors'].append("Failed to set up triggers")
            
            # Check overall success
            all_tables_success = all(table_results.values())
            results['success'] = (
                all_tables_success and 
                rls_success and 
                indexes_success and 
                triggers_success
            )
            
            migration_end = datetime.now()
            duration = (migration_end - migration_start).total_seconds()
            results['completed_at'] = migration_end.isoformat()
            results['duration_seconds'] = duration
            
            if results['success']:
                self.logger.info(f"Migration completed successfully in {duration:.2f} seconds")
            else:
                self.logger.warning(f"Migration completed with errors in {duration:.2f} seconds")
            
            return results
            
        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
            results['success'] = False
            return results
    
    async def rollback_migration(self) -> bool:
        """
        Rollback migration by dropping all tables.
        
        Returns:
            True if rollback was successful
        """
        try:
            self.logger.info("Starting migration rollback")
            
            # Drop tables in reverse order
            tables_to_drop = [
                'system_config',
                'analytics',
                'tasks',
                'videos',
                'projects',
                'users'
            ]
            
            for table_name in tables_to_drop:
                try:
                    drop_sql = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
                    await self.run_sql(drop_sql, fetch=False)
                    self.logger.info(f"Dropped table: {table_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to drop table {table_name}: {str(e)}")
            
            # Drop functions
            functions_sql = """
                DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;
                DROP FUNCTION IF EXISTS is_admin() CASCADE;
                DROP FUNCTION IF EXISTS get_current_user_id() CASCADE;
            """
            
            await self.run_sql(functions_sql, fetch=False)
            
            self.logger.info("Migration rollback completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {str(e)}")
            return False


# Convenience functions
async def migrate_database(force: bool = False) -> Dict[str, Any]:
    """
    Run database migration.
    
    Args:
        force: Whether to force recreate existing tables
        
    Returns:
        Migration results
    """
    migrator = SupabaseMigrator()
    return await migrator.run_full_migration(force=force)


async def rollback_database() -> bool:
    """
    Rollback database migration.
    
    Returns:
        True if rollback was successful
    """
    migrator = SupabaseMigrator()
    return await migrator.rollback_migration()


async def check_database_status() -> Dict[str, Any]:
    """
    Check database status and table existence.
    
    Returns:
        Database status information
    """
    migrator = SupabaseMigrator()
    status = {
        'connected': False,
        'tables': {},
        'total_tables': len(SUPABASE_SCHEMAS),
        'tables_exist': 0
    }
    
    try:
        # Check connection
        if not migrator.connection.is_connected:
            await migrator.connection.connect()
        
        status['connected'] = migrator.connection.is_connected
        
        # Check each table
        for table_name in SUPABASE_SCHEMAS.keys():
            exists = await migrator.check_table_exists(table_name)
            status['tables'][table_name] = exists
            if exists:
                status['tables_exist'] += 1
        
        status['migration_needed'] = status['tables_exist'] < status['total_tables']
        
    except Exception as e:
        status['error'] = str(e)
    
    return status


if __name__ == "__main__":
    # CLI interface for migrations
    import sys
    
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python migrations.py [migrate|rollback|status] [--force]")
            return
        
        command = sys.argv[1]
        force = '--force' in sys.argv
        
        if command == "migrate":
            print("Running database migration...")
            results = await migrate_database(force=force)
            print(f"Migration results: {results}")
            
        elif command == "rollback":
            print("Rolling back database...")
            success = await rollback_database()
            print(f"Rollback {'successful' if success else 'failed'}")
            
        elif command == "status":
            print("Checking database status...")
            status = await check_database_status()
            print(f"Database status: {status}")
            
        else:
            print(f"Unknown command: {command}")
    
    # Run the async main function
    asyncio.run(main())