# Supabase Database Integration Setup Guide

This guide will help you set up the complete Supabase database integration for MoneyPrinterTurbo.

## 🚀 Quick Start

### 1. Install Dependencies

First, install all required Python packages:

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install manually
pip install supabase asyncpg psycopg2-binary python-dotenv
```

### 2. Environment Configuration

Copy the environment template and configure your credentials:

```bash
# Copy environment template
cp .env.example .env
```

Edit `.env` with your Supabase project credentials:

```bash
# Required Supabase Settings
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-public-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-secret-key-here
```

### 3. Run Database Migration

Execute the migration to set up your database:

```bash
# Run complete migration with all components
python test_migration.py

# Or run individual components
python test_migration.py migrate    # Just migration
python test_migration.py status     # Check status only
```

## 📋 Detailed Setup Instructions

### Prerequisites

- Python 3.8 or higher
- A Supabase account and project
- pip package manager

### Step 1: Get Supabase Credentials

1. **Create a Supabase Project:**
   - Go to [supabase.com](https://supabase.com)
   - Create a new project
   - Wait for setup to complete

2. **Get API Keys:**
   - Go to Project Settings → API
   - Copy the following:
     - Project URL (`SUPABASE_URL`)
     - Project API Keys → `anon` `public` (`SUPABASE_ANON_KEY`)
     - Project API Keys → `service_role` `secret` (`SUPABASE_SERVICE_ROLE_KEY`)

3. **Configure Database:**
   - Go to Project Settings → Database
   - Note the connection string (optional for direct PostgreSQL access)

### Step 2: Environment Setup

Create and configure your environment file:

```bash
# Copy the template
cp .env.example .env

# Edit with your favorite editor
nano .env  # or vim, code, etc.
```

**Required Configuration:**
```bash
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Application Settings
APP_ENV=development
DEBUG=true
LOG_LEVEL=info
```

### Step 3: Install Dependencies

**Option A: Using requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Manual Installation**
```bash
# Core dependencies
pip install supabase==2.0.0
pip install asyncpg==0.29.0
pip install psycopg2-binary==2.9.9
pip install python-dotenv==1.0.0

# Optional: Development dependencies
pip install pytest pytest-asyncio black flake8
```

**Option C: Using Virtual Environment (Best Practice)**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Run Migration

Execute the database migration:

```bash
# Full migration (recommended for first setup)
python test_migration.py

# Check what this will do first
python test_migration.py status
```

**Migration includes:**
- ✅ Environment validation
- ✅ Database connection testing
- ✅ Table creation (users, projects, videos, tasks, analytics, system_config)
- ✅ Row Level Security (RLS) setup
- ✅ Performance indexes
- ✅ Database triggers
- ✅ Verification tests

## 🔧 Usage Examples

### Basic Connection

```python
from app.database.connection import ensure_connection

# Get connected client
conn = await ensure_connection()

# Use Supabase client directly
users = conn.client.table('users').select('*').execute()
print(f"Found {len(users.data)} users")
```

### Running Migrations

```python
from app.database.migrations import migrate_database

# Run full migration
results = await migrate_database(force=True)
print(f"Migration success: {results['success']}")
```

### Database Operations

```python
# Create a user
user_data = {
    'email': 'user@example.com',
    'name': 'John Doe',
    'role': 'user'
}
result = conn.client.table('users').insert(user_data).execute()

# Query projects
projects = conn.client.table('projects').select('*').eq('status', 'active').execute()
```

## 🧪 Testing

### Run Integration Tests

```bash
# Full test suite
python test_migration.py

# Individual tests
python test_migration.py status      # Database status
python test_migration.py migrate     # Migration only
python test_migration.py rollback    # Rollback changes
```

### Expected Test Output

```
🧪 Starting Supabase integration tests...

==================================================
TEST 1: Environment Configuration
==================================================
✅ Environment variables check passed

==================================================
TEST 2: Database Connection
==================================================
🔌 Testing Supabase connection...
✅ Supabase connection successful
   - Client available: True
   - PostgreSQL pool: True
   - API responsive: True

==================================================
TEST 3: Database Status Check
==================================================
📊 Checking database status...
   - Connected: True
   - Total tables expected: 6
   - Tables exist: 0
   - Migration needed: True

==================================================
TEST 4: Database Migration
==================================================
🚀 Testing database migration...
   - Migration success: True
   - Duration: 2.34 seconds
   Table creation results:
     ✅ users
     ✅ projects
     ✅ videos
     ✅ tasks
     ✅ analytics
     ✅ system_config
     ✅ rls_policies
     ✅ indexes
     ✅ triggers

==================================================
TEST 5: Basic Database Operations
==================================================
🔧 Testing basic database operations...
   ✅ Users table query successful
   - Query returned 0 rows

==================================================
TEST 6: Post-Migration Status
==================================================
📊 Checking database status...
   - Connected: True
   - Total tables expected: 6
   - Tables exist: 6
   - Migration needed: False

==================================================
TEST SUMMARY
==================================================
Tests passed: 6/6
🎉 All tests passed! Supabase integration is working correctly.
```

## 🔒 Security Features

The integration includes comprehensive security:

- **Row Level Security (RLS)** enabled on all tables
- **User-specific data access** policies
- **Admin role checking** functions
- **Secure credential management** via environment variables
- **Input validation** and sanitization

## 📊 Database Schema

The migration creates these tables:

| Table | Purpose | Key Features |
|-------|---------|--------------|
| `users` | User accounts and authentication | UUID primary key, role-based access |
| `projects` | Video projects and metadata | User ownership, status tracking |
| `videos` | Generated videos and status | Project association, file paths |
| `tasks` | Background job queue | Priority queue, status tracking |
| `analytics` | Usage tracking and metrics | Event logging, user analytics |
| `system_config` | Application configuration | Key-value configuration store |

## 🚨 Troubleshooting

### Common Issues

**1. Module Not Found Error (asyncpg)**
```bash
# Install missing dependencies
pip install asyncpg psycopg2-binary

# Or install all dependencies
pip install -r requirements.txt
```

**2. Environment Variables Not Found**
```bash
# Make sure .env file exists and has correct format
cp .env.example .env
# Edit .env with your Supabase credentials
```

**3. Connection Failed**
```bash
# Verify your Supabase credentials
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('URL:', os.getenv('SUPABASE_URL'))
print('Key:', os.getenv('SUPABASE_ANON_KEY')[:20] + '...')
"
```

**4. Permission Denied**
```bash
# Make sure you're using the service role key for migrations
# Check that SUPABASE_SERVICE_ROLE_KEY is set correctly
```

### Debug Mode

Enable debug logging for more information:

```bash
# Set debug mode in .env
DEBUG=true
LOG_LEVEL=debug

# Run with verbose output
python test_migration.py
```

### Reset Database

If you need to start over:

```bash
# Rollback all changes
python test_migration.py rollback

# Run fresh migration
python test_migration.py migrate
```

## 🔄 Development Workflow

### Making Schema Changes

1. **Update Models** (`app/database/models.py`)
2. **Update Migrations** (`app/database/migrations.py`)
3. **Test Changes** (`python test_migration.py`)
4. **Update Documentation**

### Adding New Tables

1. Add schema to `SUPABASE_SCHEMAS` in `models.py`
2. Add table creation order in `migrations.py`
3. Run migration to test
4. Update this documentation

## 📚 Next Steps

Once setup is complete, you can:

1. **Integrate with your application** using the connection manager
2. **Add business logic** using the database models
3. **Monitor performance** with the analytics table
4. **Scale up** with the existing infrastructure
5. **Add new features** following the established patterns

## 🆘 Support

If you encounter issues:

1. Check the [troubleshooting section](#🚨-troubleshooting)
2. Run `python test_migration.py status` to diagnose problems
3. Check Supabase dashboard for connection issues
4. Verify environment variables are correctly set
5. Ensure all dependencies are installed

The integration is designed to be robust and provide clear error messages to help diagnose and fix issues quickly.