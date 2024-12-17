import sqlite3
import json
from datetime import datetime
from functools import wraps
from typing import Dict, List, Tuple, Optional

def safe_db_operation(operation):
    @wraps(operation)
    def wrapper(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except sqlite3.Error as e:
            return f"Database error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    return wrapper

class SnapshotDB:
    def __init__(self, db_path: str = 'prompts_snapshots.db'):
        self.db_path = db_path
        self.init_db()

    @safe_db_operation
    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS snapshots
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         snapshot_name TEXT NOT NULL,
                         user_prompt TEXT NOT NULL,
                         system_prompt TEXT,
                         model_name TEXT NOT NULL,
                         cot_prompt TEXT,
                         initial_response TEXT,
                         thinking TEXT,
                         reflection TEXT,
                         final_response TEXT,
                         created_at TIMESTAMP,
                         tags TEXT)''')
            conn.commit()

    @safe_db_operation
    def save_snapshot(self, snapshot_data: Dict) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''INSERT INTO snapshots
                            (snapshot_name, user_prompt, system_prompt, model_name, 
                             cot_prompt, initial_response, thinking, reflection, 
                             final_response, created_at, tags)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (snapshot_data['snapshot_name'],
                          snapshot_data['user_prompt'],
                          snapshot_data['system_prompt'],
                          snapshot_data['model_name'],
                          snapshot_data['cot_prompt'],
                          snapshot_data['initial_response'],
                          snapshot_data['thinking'],
                          snapshot_data['reflection'],
                          snapshot_data['final_response'],
                          datetime.now(),
                          snapshot_data.get('tags', '')))
                conn.commit()
                return "Success"
        except sqlite3.Error as e:
            return f"Database error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    @safe_db_operation
    def get_snapshots(self, search_term: str = None) -> List[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            if search_term:
                query = '''SELECT * FROM snapshots 
                          WHERE snapshot_name LIKE ? 
                          OR user_prompt LIKE ? 
                          OR tags LIKE ?
                          ORDER BY created_at DESC'''
                search_pattern = f'%{search_term}%'
                c.execute(query, (search_pattern, search_pattern, search_pattern))
            else:
                c.execute('SELECT * FROM snapshots ORDER BY created_at DESC')
            return c.fetchall()

    @safe_db_operation
    def get_snapshot_by_id(self, snapshot_id: int) -> Optional[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM snapshots WHERE id = ?', (snapshot_id,))
            return c.fetchone()

    @safe_db_operation
    def delete_snapshot(self, snapshot_id: int) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('DELETE FROM snapshots WHERE id = ?', (snapshot_id,))
                if c.rowcount > 0:
                    conn.commit()
                    return "Success"
                return "Snapshot not found"
        except sqlite3.Error as e:
            return f"Database error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    @safe_db_operation
    def export_snapshots(self, format: str = 'json') -> str:
        snapshots = self.get_snapshots()
        if format == 'json':
            snapshot_list = []
            for s in snapshots:
                snapshot_dict = {
                    'id': s[0],
                    'name': s[1],
                    'user_prompt': s[2],
                    'system_prompt': s[3],
                    'model_name': s[4],
                    'cot_prompt': s[5],
                    'initial_response': s[6],
                    'thinking': s[7],
                    'reflection': s[8],
                    'final_response': s[9],
                    'created_at': str(s[10]),
                    'tags': s[11]
                }
                snapshot_list.append(snapshot_dict)
            return json.dumps(snapshot_list, indent=2)
        return "Unsupported export format"