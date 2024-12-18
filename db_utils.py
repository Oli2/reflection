import sqlite3
import json
from datetime import datetime
from functools import wraps
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

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

@dataclass
class SnapshotData:
    """Data model for snapshot information."""
    id: int
    name: str
    model_name: str
    user_prompt: str
    system_prompt: Optional[str] = None
    cot_prompt: Optional[str] = None
    initial_response: Optional[str] = None
    thinking: Optional[str] = None
    reflection: Optional[str] = None
    final_response: Optional[str] = None
    created_at: Optional[datetime] = None
    tags: Optional[str] = None

    @classmethod
    def from_db_row(cls, row: tuple) -> 'SnapshotData':
        """Create SnapshotData instance from database row."""
        return cls(
            id=row[0],
            name=row[1],
            user_prompt=row[2],
            system_prompt=row[3],
            model_name=row[4],
            cot_prompt=row[5],
            initial_response=row[6],
            thinking=row[7],
            reflection=row[8],
            final_response=row[9],
            created_at=row[10],
            tags=row[11]
        )

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
        """
        Save snapshot to database.
        
        Args:
            snapshot_data: Dictionary containing snapshot data
            
        Returns:
            Status message
        """
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
                return "✓ Snapshot saved successfully"
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
    def get_snapshot_by_id(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a snapshot by ID and return as a dictionary.
        
        Args:
            snapshot_id: The ID of the snapshot to retrieve
            
        Returns:
            Dictionary containing snapshot data if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('SELECT * FROM snapshots WHERE id = ?', (snapshot_id,))
                snapshot = c.fetchone()
                
                if not snapshot:
                    return None
                
                # Convert snapshot data to dictionary
                return {
                    "snapshot_name": snapshot[1],
                    "user_prompt": snapshot[2],
                    "system_prompt": snapshot[3],
                    "model_name": snapshot[4],
                    "cot_prompt": snapshot[5],
                    "initial_response": snapshot[6],
                    "thinking": snapshot[7],
                    "reflection": snapshot[8],
                    "final_response": snapshot[9],
                    "created_at": snapshot[10],
                    "tags": snapshot[11]
                }
                
        except Exception as e:
            print(f"Database retrieval error: {e}")
            return None

    @safe_db_operation
    def delete_selected_snapshots(self, selected_rows: List[List]) -> Tuple[str, List[List]]:
        """Delete selected snapshots and return updated table data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                for row in selected_rows:
                    snapshot_id = row[0]  # First column is ID
                    c.execute('DELETE FROM snapshots WHERE id = ?', (snapshot_id,))
                conn.commit()
                return "✓ Selected snapshots deleted successfully", self.get_snapshots()
        except Exception as e:
            return f"Error deleting snapshots: {str(e)}", self.get_snapshots()

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