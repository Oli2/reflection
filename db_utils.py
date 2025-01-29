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
    def __init__(self, db_path="prompts_snapshots.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize database with both snapshots and evaluations tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Existing snapshots table remains unchanged
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_name TEXT,
                    user_prompt TEXT,
                    system_prompt TEXT,
                    model_name TEXT,
                    cot_prompt TEXT,
                    initial_response TEXT,
                    thinking TEXT,
                    reflection TEXT,
                    final_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT
                )
            ''')
            
            # Add new evaluations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot1_id INTEGER,
                    snapshot2_id INTEGER,
                    judge_model TEXT,
                    compared_aspects TEXT,  -- JSON string of selected aspects
                    evaluation_criteria TEXT,  -- JSON string of customized criteria
                    numerical_scores TEXT,  -- JSON string of scores
                    qualitative_analysis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (snapshot1_id) REFERENCES snapshots(id),
                    FOREIGN KEY (snapshot2_id) REFERENCES snapshots(id)
                )
            ''')
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

    def get_snapshots(self, search_term: str = "") -> List[List]:
        """Get all snapshots, optionally filtered by search term."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if search_term:
                    query = """
                        SELECT 
                            id,             -- 0
                            snapshot_name,   -- 1
                            created_at,      -- 2
                            model_name,      -- 3
                            user_prompt,     -- 4
                            tags            -- 5
                        FROM snapshots
                        WHERE snapshot_name LIKE ? 
                           OR user_prompt LIKE ? 
                           OR tags LIKE ?
                        ORDER BY created_at DESC
                    """
                    search_pattern = f"%{search_term}%"
                    cursor.execute(query, (search_pattern, search_pattern, search_pattern))
                else:
                    query = """
                        SELECT 
                            id,             -- 0
                            snapshot_name,   -- 1
                            created_at,      -- 2
                            model_name,      -- 3
                            user_prompt,     -- 4
                            tags            -- 5
                        FROM snapshots
                        ORDER BY created_at DESC
                    """
                    cursor.execute(query)
                
                return cursor.fetchall()
                
        except sqlite3.Error as e:
            print(f"Database error: {str(e)}")
            return []
        except Exception as e:
            print(f"Error getting snapshots: {str(e)}")
            return []

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

    def delete_snapshot(self, snapshot_id: int) -> Tuple[str, List[List]]:
        """Delete a snapshot by its ID and return updated table data."""
        try:
            if not isinstance(snapshot_id, (int, float)) or snapshot_id <= 0:
                return "Invalid snapshot ID", self.get_snapshots()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if snapshot exists
                cursor.execute("SELECT id FROM snapshots WHERE id = ?", (snapshot_id,))
                if not cursor.fetchone():
                    return f"Snapshot with ID {snapshot_id} not found", self.get_snapshots()
                
                # Delete the snapshot
                cursor.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
                conn.commit()
                
                # Get updated table data
                return f"Snapshot {snapshot_id} deleted successfully", self.get_snapshots()
                
        except sqlite3.Error as e:
            return f"Database error: {str(e)}", self.get_snapshots()
        except Exception as e:
            return f"Error deleting snapshot: {str(e)}", self.get_snapshots()

    def delete_selected_snapshots(self, selected_data: List[List]) -> Tuple[str, List[List]]:
        """
        Delete selected snapshots from the database.
        
        Args:
            selected_data: List of selected rows from the Gradio Dataframe
            
        Returns:
            Tuple of (status_message: str, updated_table_data: List[List])
        """
        if not selected_data or not isinstance(selected_data, list):
            return "No snapshots selected", self.get_snapshots()
            
        try:
            deleted_count = 0
            for row in selected_data:
                if row and len(row) > 0:
                    snapshot_id = row[0]  # First column is ID
                    status, updated_data = self.delete_snapshot(snapshot_id)
                    if status.startswith("Successfully"):
                        deleted_count += 1
            
            return f"Successfully deleted {deleted_count} snapshot(s)", updated_data
            
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

    # Add new methods for evaluation operations
    def save_evaluation(self, evaluation_data: Dict) -> Tuple[bool, str]:
        """Save a new evaluation"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO evaluations (
                        snapshot1_id,
                        snapshot2_id,
                        judge_model,
                        compared_aspects,
                        evaluation_criteria,
                        numerical_scores,
                        qualitative_analysis
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    evaluation_data['snapshot1_id'],
                    evaluation_data['snapshot2_id'],
                    evaluation_data['judge_model'],
                    json.dumps(evaluation_data['compared_aspects']),
                    json.dumps(evaluation_data['evaluation_criteria']),
                    json.dumps(evaluation_data['numerical_scores']),
                    evaluation_data['qualitative_analysis']
                ))
                conn.commit()
                return True, "Evaluation saved successfully"
        except Exception as e:
            return False, f"Error saving evaluation: {str(e)}"

    def get_evaluation_by_id(self, evaluation_id: int) -> Optional[Dict]:
        """Retrieve an evaluation by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM evaluations WHERE id = ?
                ''', (evaluation_id,))
                row = cursor.fetchone()
                
                if row:
                    evaluation = dict(row)
                    # Parse JSON strings back to Python objects
                    evaluation['compared_aspects'] = json.loads(evaluation['compared_aspects'])
                    evaluation['evaluation_criteria'] = json.loads(evaluation['evaluation_criteria'])
                    evaluation['numerical_scores'] = json.loads(evaluation['numerical_scores'])
                    return evaluation
                return None
        except Exception as e:
            print(f"Error retrieving evaluation: {str(e)}")
            return None

    def get_evaluations_for_snapshot(self, snapshot_id: int) -> List[Dict]:
        """Get all evaluations involving a specific snapshot"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM evaluations 
                    WHERE snapshot1_id = ? OR snapshot2_id = ?
                    ORDER BY created_at DESC
                ''', (snapshot_id, snapshot_id))
                
                evaluations = []
                for row in cursor.fetchall():
                    evaluation = dict(row)
                    evaluation['compared_aspects'] = json.loads(evaluation['compared_aspects'])
                    evaluation['evaluation_criteria'] = json.loads(evaluation['evaluation_criteria'])
                    evaluation['numerical_scores'] = json.loads(evaluation['numerical_scores'])
                    evaluations.append(evaluation)
                return evaluations
        except Exception as e:
            print(f"Error retrieving evaluations: {str(e)}")
            return []

    def get_recent_evaluations(self, limit: int = 5) -> List[Dict]:
        """Get most recent evaluations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT e.*, 
                           s1.snapshot_name as snapshot1_name,
                           s2.snapshot_name as snapshot2_name
                    FROM evaluations e
                    JOIN snapshots s1 ON e.snapshot1_id = s1.id
                    JOIN snapshots s2 ON e.snapshot2_id = s2.id
                    ORDER BY e.created_at DESC
                    LIMIT ?
                ''', (limit,))
                
                evaluations = []
                for row in cursor.fetchall():
                    evaluation = dict(row)
                    evaluation['compared_aspects'] = json.loads(evaluation['compared_aspects'])
                    evaluation['evaluation_criteria'] = json.loads(evaluation['evaluation_criteria'])
                    evaluation['numerical_scores'] = json.loads(evaluation['numerical_scores'])
                    evaluations.append(evaluation)
                return evaluations
        except Exception as e:
            print(f"Error retrieving recent evaluations: {str(e)}")
            return []