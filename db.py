import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Union, Dict, List
import os
import uuid

DB_NAME = 'your_database'
DB_USER = 'your_username'
DB_PASSWORD = 'your_password'
DB_HOST = 'localhost'  # or your database server IP
DB_PORT = '5432'       # default PostgreSQL port

def init_db():
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename TEXT,
            filepath TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id SERIAL PRIMARY KEY,
            query TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS response (
            id UUID PRIMARY KEY,
            answer TEXT,
            bullet_points TEXT,
            test_question TEXT,
            test_answer TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id SERIAL PRIMARY KEY,
            test_question_id UUID,
            user_answer TEXT,
            knowledge_understood BOOLEAN,
            knowledge_confidence INTEGER,
            FOREIGN KEY (test_question_id) REFERENCES response (id)
        )
    ''')

    conn.commit()
    print("Database initialized with tables.")
    cursor.close()
    conn.close()

# Call init_db() when this module is imported
init_db()

def get_db_connection():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    return conn

def store_document_metadata(filename: str, filepath: str) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents (filename, filepath) VALUES (%s, %s)", (filename, filepath))
    conn.commit()
    cursor.close()
    conn.close()

def store_query_metadata(query: str) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO queries (query) VALUES (%s)", (query,))
    conn.commit()
    cursor.close()
    conn.close()

def store_response_metadata(answer: str, bullet_points: str, test_question: str, test_answer: str) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    response_id = uuid.uuid4()
    cursor.execute(
        "INSERT INTO response (id, answer, bullet_points, test_question, test_answer) VALUES (%s, %s, %s, %s, %s)", (
            str(response_id), answer, bullet_points, test_question, test_answer))
    conn.commit()
    cursor.close()
    conn.close()
    return str(response_id)

def store_knowledge_evaluation_metadata(test_question_id: str, 
user_answer: str, 
knowledge_understood: bool, 
knowledge_confidence: int) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO evaluations (test_question_id, user_answer, knowledge_understood, knowledge_confidence) VALUES (%s, %s, %s, %s)",
                   (str(test_question_id), user_answer, knowledge_understood, knowledge_confidence))
    conn.commit()
    cursor.close()
    conn.close()

def get_document_metadata() -> List[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM documents")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [dict(row) for row in rows]  # Ensure conversion to standard dict

def get_query_metadata() -> List[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM queries")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [dict(row) for row in rows]  # Ensure conversion to standard dict

def get_response_metadata(response_id: Optional[str] = None) -> Union[Optional[Dict], List[Dict]]:
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    if response_id:
        cursor.execute("SELECT * FROM response WHERE id = %s", (response_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row if row else None
    else:
        cursor.execute("SELECT * FROM response")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [dict(row) for row in rows]  # Ensure conversion to standard dict


def get_evaluation_metadata() -> List[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM evaluations")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [dict(row) for row in rows]
