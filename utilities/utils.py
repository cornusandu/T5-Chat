from abc import ABC, abstractmethod
import sqlite3

#  Base Class. Inheritance is not recommended.
class SQL_Class(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def save_sql(self, conn: sqlite3.Connection):
        """
        Save the instance data to the database using the class name as the table name.
        Works with inheritance by dynamically creating tables based on child class attributes.
        
        Args:
            conn (sqlite3.Connection): SQLite database connection
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # cursor = conn.cursor()
            
            # Get class name for table name (lowercase)
            table_name = self.__class__.__name__.lower()
            
            # Get instance attributes (excluding private/special attributes)
            attributes = {k: v for k, v in self.__dict__.items() 
                         if not k.startswith('_')}
            
            if not attributes:
                print(f"Warning: No attributes found for {self.__class__.__name__}")
                return False
                
            # Create table if it doesn't exist with columns for each attribute
            columns = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
            columns.extend([f"{attr} TEXT" for attr in attributes.keys()])
            columns.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {', '.join(columns)}
                )
            """
            conn.execute(create_table_sql)
            
            # Insert data from this instance
            placeholders = ', '.join(['?'] * len(attributes))
            column_names = ', '.join(attributes.keys())
            
            insert_sql = f"""
                INSERT INTO {table_name} ({column_names})
                VALUES ({placeholders})
            """
            
            conn.execute(insert_sql, tuple(attributes.values()))
            
            # Commit the transaction
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            # Rollback in case of error
            conn.rollback()
            print(f"Database error: {e}")
            return False
        except Exception as e:
            # Rollback in case of error
            conn.rollback()
            print(f"Error: {e}")
            return False

class Users(SQL_Class):
    def __init__(self, id: int, username: str, password: str):
        self.id = id
        self.username = username
        self.display_name = username
        self.password = password
        self.description = ""
        self.instructions = ""

    def save_sql(self, conn: sqlite3.Connection):
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, display_name TEXT, description TEXT, instructions TEXT)")
        conn.execute("INSERT INTO users (id, username, password, display_name, description, instructions) VALUES (?, ?, ?, ?, ?, ?)", (self.id, self.username, self.password, self.display_name, self.description, self.instructions))
        conn.commit()
