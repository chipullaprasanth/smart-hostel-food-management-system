import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

# Try to import mysql connector, fail gracefully to sqlite
USE_MYSQL = False
MYSQL_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASS', ''),
    'database': os.environ.get('DB_NAME', 'hostel_db')
}

try:
    import pymysql
    # Test connection
    conn = pymysql.connect(
        host=MYSQL_CONFIG['host'],
        user=MYSQL_CONFIG['user'],
        password=MYSQL_CONFIG['password'],
        timeout=2
    )
    conn.close()
    USE_MYSQL = True
    print("Database Layer: MySQL detected and connected successfully.")
except Exception as e:
    print(f"Database Layer: MySQL connection failed or packages missing ({e}). Falling back to local SQLite.")

DB_PATH = "food_data.db"

def get_connection():
    if USE_MYSQL:
        conn = pymysql.connect(
            host=MYSQL_CONFIG['host'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            database=MYSQL_CONFIG['database'],
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def init_db():
    if USE_MYSQL:
        # Create database first
        conn = pymysql.connect(
            host=MYSQL_CONFIG['host'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password']
        )
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
        conn.commit()
        conn.close()

    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Helper to execute queries on both engines
        def run_ddl(sqlite_ddl, mysql_ddl):
            if USE_MYSQL:
                cursor.execute(mysql_ddl)
            else:
                cursor.execute(sqlite_ddl)
                
        # Predictions table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day TEXT,
            month TEXT,
            temperature REAL,
            rainfall REAL,
            humidity INTEGER,
            wind_speed REAL,
            breakfast_menu TEXT,
            lunch_menu TEXT,
            dinner_menu TEXT,
            previous_attendance INTEGER,
            previous_waste REAL,
            food_rating REAL,
            predicted_attendance INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """, """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            day VARCHAR(15),
            month VARCHAR(15),
            temperature FLOAT,
            rainfall FLOAT,
            humidity INT,
            wind_speed FLOAT,
            breakfast_menu VARCHAR(50),
            lunch_menu VARCHAR(50),
            dinner_menu VARCHAR(50),
            previous_attendance INT,
            previous_waste FLOAT,
            food_rating FLOAT,
            predicted_attendance INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Users table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
        """, """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE,
            password VARCHAR(255)
        )
        """)
        
        # Feedback table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            feedback TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """, """
        CREATE TABLE IF NOT EXISTS feedback (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50),
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Attendance table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            students INTEGER
        )
        """, """
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date VARCHAR(20) UNIQUE,
            students INT
        )
        """)
        
        # Inventory table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item TEXT UNIQUE,
            quantity TEXT
        )
        """, """
        CREATE TABLE IF NOT EXISTS inventory (
            id INT AUTO_INCREMENT PRIMARY KEY,
            item VARCHAR(50) UNIQUE,
            quantity VARCHAR(30)
        )
        """)
        
        # Donations table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS donations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization TEXT,
            food TEXT,
            quantity TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """, """
        CREATE TABLE IF NOT EXISTS donations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            organization VARCHAR(100),
            food VARCHAR(100),
            quantity VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Expenses table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            amount REAL,
            description TEXT
        )
        """, """
        CREATE TABLE IF NOT EXISTS expenses (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date VARCHAR(20),
            amount DOUBLE,
            description VARCHAR(255)
        )
        """)

        # Waste records table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS waste_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            prepared REAL,
            consumed REAL,
            waste REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """, """
        CREATE TABLE IF NOT EXISTS waste_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date VARCHAR(20) UNIQUE,
            prepared FLOAT,
            consumed FLOAT,
            waste FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
    
    # Initialize default admin user if not exists
    register_user("admin", "admin123")
    
    # Seed default inventory if empty
    seed_default_inventory()

def register_user(username, password):
    hashed = generate_password_hash(password)
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)" if USE_MYSQL else "INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
            return True
    except Exception:
        return False

def verify_user(username, password):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s" if USE_MYSQL else "SELECT password FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            stored_pwd = row['password'] if USE_MYSQL else row[0]
            if check_password_hash(stored_pwd, password):
                return True
    return False

def add_prediction(data_dict):
    sql = """
    INSERT INTO predictions (
        day, month, temperature, rainfall, humidity, wind_speed,
        breakfast_menu, lunch_menu, dinner_menu,
        previous_attendance, previous_waste, food_rating, predicted_attendance
    ) VALUES ({param}, {param}, {param}, {param}, {param}, {param}, {param}, {param}, {param}, {param}, {param}, {param}, {param})
    """.format(param="%s" if USE_MYSQL else "?")
    
    values = (
        data_dict['Day'], data_dict['Month'], data_dict['Temperature'], data_dict['Rainfall'],
        data_dict['Humidity'], data_dict['Wind_Speed'], data_dict['Breakfast_Menu'], data_dict['Lunch_Menu'],
        data_dict['Dinner_Menu'], data_dict['Previous_Attendance'], data_dict['Previous_Waste'],
        data_dict['Food_Rating'], data_dict['Predicted_Attendance']
    )
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, values)
        conn.commit()

def get_predictions(search=None):
    with get_connection() as conn:
        cursor = conn.cursor()
        if search:
            q = f"SELECT * FROM predictions WHERE day LIKE %s ORDER BY id DESC" if USE_MYSQL else f"SELECT * FROM predictions WHERE day LIKE ? ORDER BY id DESC"
            cursor.execute(q, (f"%{search}%",))
        else:
            cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
        rows = cursor.fetchall()
        
        # Normalize result rows to standard dictionaries
        results = []
        for r in rows:
            if USE_MYSQL:
                results.append(dict(r))
            else:
                results.append({
                    'id': r[0], 'day': r[1], 'month': r[2], 'temperature': r[3], 'rainfall': r[4],
                    'humidity': r[5], 'wind_speed': r[6], 'breakfast_menu': r[7], 'lunch_menu': r[8],
                    'dinner_menu': r[9], 'previous_attendance': r[10], 'previous_waste': r[11],
                    'food_rating': r[12], 'predicted_attendance': r[13], 'created_at': r[14]
                })
        return results

def delete_prediction(prediction_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = %s" if USE_MYSQL else "DELETE FROM predictions WHERE id = ?", (prediction_id,))
        conn.commit()

def add_feedback(username, text):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO feedback (username, feedback) VALUES (%s, %s)" if USE_MYSQL else "INSERT INTO feedback (username, feedback) VALUES (?, ?)", (username, text))
        conn.commit()

def get_feedback_count():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feedback")
        row = cursor.fetchone()
        return row['COUNT(*)'] if USE_MYSQL else row[0]

def get_predictions_count():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        row = cursor.fetchone()
        return row['COUNT(*)'] if USE_MYSQL else row[0]

def get_users_count():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        row = cursor.fetchone()
        return row['COUNT(*)'] if USE_MYSQL else row[0]

def save_attendance(date, students):
    with get_connection() as conn:
        cursor = conn.cursor()
        if USE_MYSQL:
            cursor.execute(
                "INSERT INTO attendance (date, students) VALUES (%s, %s) ON DUPLICATE KEY UPDATE students=VALUES(students)",
                (date, students)
            )
        else:
            cursor.execute(
                "INSERT INTO attendance (date, students) VALUES (?, ?) ON CONFLICT(date) DO UPDATE SET students=excluded.students",
                (date, students)
            )
        conn.commit()

def seed_default_inventory():
    default_items = [
        ("Rice", "150 Kg"),
        ("Dal", "50 Kg"),
        ("Vegetables", "80 Kg"),
        ("Cooking Oil", "25 L"),
        ("Spices", "15 Kg")
    ]
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inventory")
            cnt = cursor.fetchone()
            count_val = cnt['COUNT(*)'] if USE_MYSQL else cnt[0]
            if count_val == 0:
                stmt = "INSERT IGNORE INTO inventory (item, quantity) VALUES (%s, %s)" if USE_MYSQL else "INSERT OR IGNORE INTO inventory (item, quantity) VALUES (?, ?)"
                cursor.executemany(stmt, default_items)
                conn.commit()
    except Exception:
        pass

def save_inventory(item, quantity):
    with get_connection() as conn:
        cursor = conn.cursor()
        if USE_MYSQL:
            cursor.execute(
                "INSERT INTO inventory (item, quantity) VALUES (%s, %s) ON DUPLICATE KEY UPDATE quantity=VALUES(quantity)",
                (item, quantity)
            )
        else:
            cursor.execute(
                "INSERT INTO inventory (item, quantity) VALUES (?, ?) ON CONFLICT(item) DO UPDATE SET quantity=excluded.quantity",
                (item, quantity)
            )
        conn.commit()

def get_inventory():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT item, quantity FROM inventory")
        rows = cursor.fetchall()
        if USE_MYSQL:
            return [dict(r) for r in rows]
        else:
            return [{'item': r[0], 'quantity': r[1]} for r in rows]

def save_donation(organization, food, quantity):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO donations (organization, food, quantity) VALUES (%s, %s, %s)" if USE_MYSQL else "INSERT INTO donations (organization, food, quantity) VALUES (?, ?, ?)", (organization, food, quantity))
        conn.commit()

def save_expense(date, amount, description):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO expenses (date, amount, description) VALUES (%s, %s, %s)" if USE_MYSQL else "INSERT INTO expenses (date, amount, description) VALUES (?, ?, ?)", (date, amount, description))
        conn.commit()

def get_expense_stats():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(amount) FROM expenses")
        row = cursor.fetchone()
        val = row['SUM(amount)'] if USE_MYSQL else row[0]
        total_expense = val if val is not None else 0.0
        monthly_savings = max(4000.0, 30000.0 - total_expense)
        return total_expense, monthly_savings

def save_waste_record(date, prepared, consumed, waste):
    with get_connection() as conn:
        cursor = conn.cursor()
        if USE_MYSQL:
            cursor.execute(
                "INSERT INTO waste_records (date, prepared, consumed, waste) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE prepared=VALUES(prepared), consumed=VALUES(consumed), waste=VALUES(waste)",
                (date, prepared, consumed, waste)
            )
        else:
            cursor.execute(
                "INSERT INTO waste_records (date, prepared, consumed, waste) VALUES (?, ?, ?, ?) ON CONFLICT(date) DO UPDATE SET prepared=excluded.prepared, consumed=excluded.consumed, waste=excluded.waste",
                (date, prepared, consumed, waste)
            )
        conn.commit()

def get_waste_records():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT date, prepared, consumed, waste FROM waste_records ORDER BY date DESC LIMIT 15")
        rows = cursor.fetchall()
        if USE_MYSQL:
            return [dict(r) for r in rows]
        else:
            return [{'date': r[0], 'prepared': r[1], 'consumed': r[2], 'waste': r[3]} for r in rows]

def get_feedback():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, feedback, created_at FROM feedback ORDER BY id DESC")
        rows = cursor.fetchall()
        if USE_MYSQL:
            return [dict(r) for r in rows]
        else:
            return [{'username': r[0], 'feedback': r[1], 'created_at': r[2]} for r in rows]

def get_donations():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT organization, food, quantity, created_at FROM donations ORDER BY id DESC")
        rows = cursor.fetchall()
        if USE_MYSQL:
            return [dict(r) for r in rows]
        else:
            return [{'organization': r[0], 'food': r[1], 'quantity': r[2], 'created_at': r[3]} for r in rows]

