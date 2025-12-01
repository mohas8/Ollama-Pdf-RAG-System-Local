"""
Simple authentication module for student/teacher login system.
Uses SQLite for user storage with password hashing.
"""

import sqlite3
import hashlib
import secrets
import os
from datetime import datetime, timedelta
from functools import wraps
from flask import session, redirect, url_for, request, jsonify

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with user tables."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('student', 'teacher')),
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Create sessions table for token-based auth
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create exam_questions table to store generated questions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exam_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            question_type TEXT NOT NULL,
            options TEXT,
            correct_answer TEXT NOT NULL,
            difficulty TEXT,
            module TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash."""
    try:
        salt, password_hash = stored_hash.split(':')
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except:
        return False

def register_user(username: str, email: str, password: str, role: str, full_name: str = None) -> dict:
    """Register a new user."""
    if role not in ['student', 'teacher']:
        return {'success': False, 'error': 'Invalid role. Must be student or teacher.'}
    
    if len(password) < 6:
        return {'success': False, 'error': 'Password must be at least 6 characters.'}
    
    if len(username) < 3:
        return {'success': False, 'error': 'Username must be at least 3 characters.'}
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role, full_name)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, password_hash, role, full_name))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {'success': True, 'user_id': user_id, 'message': 'Registration successful!'}
    except sqlite3.IntegrityError as e:
        conn.close()
        if 'username' in str(e):
            return {'success': False, 'error': 'Username already exists.'}
        elif 'email' in str(e):
            return {'success': False, 'error': 'Email already registered.'}
        return {'success': False, 'error': 'Registration failed.'}
    except Exception as e:
        conn.close()
        return {'success': False, 'error': str(e)}

def login_user(username: str, password: str) -> dict:
    """Login user and create session token."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username, username))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        return {'success': False, 'error': 'User not found.'}
    
    if not verify_password(password, user['password_hash']):
        conn.close()
        return {'success': False, 'error': 'Invalid password.'}
    
    # Create session token
    token = secrets.token_hex(32)
    expires_at = datetime.now() + timedelta(days=7)
    
    cursor.execute('''
        INSERT INTO sessions (user_id, token, expires_at)
        VALUES (?, ?, ?)
    ''', (user['id'], token, expires_at))
    
    # Update last login
    cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.now(), user['id']))
    
    conn.commit()
    conn.close()
    
    return {
        'success': True,
        'token': token,
        'user': {
            'id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'role': user['role'],
            'full_name': user['full_name']
        }
    }

def verify_token(token: str) -> dict:
    """Verify session token and return user info."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT u.*, s.expires_at 
        FROM sessions s 
        JOIN users u ON s.user_id = u.id 
        WHERE s.token = ?
    ''', (token,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return {'valid': False, 'error': 'Invalid token.'}
    
    if datetime.strptime(result['expires_at'], '%Y-%m-%d %H:%M:%S.%f') < datetime.now():
        return {'valid': False, 'error': 'Token expired.'}
    
    return {
        'valid': True,
        'user': {
            'id': result['id'],
            'username': result['username'],
            'email': result['email'],
            'role': result['role'],
            'full_name': result['full_name']
        }
    }

def logout_user(token: str) -> dict:
    """Logout user by deleting session token."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM sessions WHERE token = ?', (token,))
    conn.commit()
    conn.close()
    return {'success': True, 'message': 'Logged out successfully.'}

def get_user_by_id(user_id: int) -> dict:
    """Get user by ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email, role, full_name, created_at FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None

def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            token = request.cookies.get('auth_token')
        
        if not token:
            return jsonify({'error': 'Authentication required', 'success': False}), 401
        
        result = verify_token(token)
        if not result['valid']:
            return jsonify({'error': result['error'], 'success': False}), 401
        
        # Add user to request context
        request.current_user = result['user']
        return f(*args, **kwargs)
    return decorated_function

def require_teacher(f):
    """Decorator to require teacher role."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            token = request.cookies.get('auth_token')
        
        if not token:
            return jsonify({'error': 'Authentication required', 'success': False}), 401
        
        result = verify_token(token)
        if not result['valid']:
            return jsonify({'error': result['error'], 'success': False}), 401
        
        if result['user']['role'] != 'teacher':
            return jsonify({'error': 'Teacher access required', 'success': False}), 403
        
        request.current_user = result['user']
        return f(*args, **kwargs)
    return decorated_function

# Initialize database on module import
init_db()
