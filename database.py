import sqlite3
import os
import pandas as pd
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
    conn = pymysql.connect(
        host=MYSQL_CONFIG['host'],
        user=MYSQL_CONFIG['user'],
        password=MYSQL_CONFIG['password'],
        connect_timeout=2
    )
    conn.close()
    USE_MYSQL = True
    print("Database Layer: MySQL detected and connected successfully.")
except Exception as e:
    print(f"Database Layer: MySQL connection skipped or fallback to local SQLite ({e}).")

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

def migrate_sqlite_columns(cursor):
    """Ensures existing SQLite database tables have user_id, role, and recommendation columns, and removes legacy unique constraints."""
    if USE_MYSQL:
        return
    try:
        # Migrate users table for role and created_at columns
        cursor.execute("PRAGMA table_info(users)")
        u_cols = [row[1] for row in cursor.fetchall()]
        if u_cols and "role" not in u_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
        if u_cols and "created_at" not in u_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")

        tables_to_add_user_id = ["predictions", "attendance", "waste_records", "expenses", "inventory", "feedback", "donations"]
        for tbl in tables_to_add_user_id:
            try:
                cursor.execute(f"PRAGMA table_info({tbl})")
                existing_cols = [row[1] for row in cursor.fetchall()]
                if existing_cols and "user_id" not in existing_cols:
                    cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN user_id INTEGER DEFAULT 1")
            except Exception:
                pass
                
        cursor.execute("PRAGMA table_info(predictions)")
        pred_cols = [row[1] for row in cursor.fetchall()]
        new_cols = [
            ("date_str", "TEXT"),
            ("festival", "TEXT"),
            ("rice_kg", "REAL"),
            ("dal_kg", "REAL"),
            ("curry_kg", "REAL"),
            ("chapati_count", "INTEGER")
        ]
        for col_name, col_type in new_cols:
            if col_name not in pred_cols:
                cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")

        cursor.execute("PRAGMA table_info(waste_records)")
        w_cols = [row[1] for row in cursor.fetchall()]
        for col_name in ["rice_waste", "dal_waste", "vegetable_waste", "chapati_waste"]:
            if w_cols and col_name not in w_cols:
                try:
                    cursor.execute(f"ALTER TABLE waste_records ADD COLUMN {col_name} REAL DEFAULT 0")
                except Exception as ex:
                    print(f"Migration note ({col_name}): {ex}")

        # Rebuild SQLite tables that contain legacy single-column UNIQUE constraints
        for tbl, sql_create, cols_list in [
            ("attendance", "CREATE TABLE attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER DEFAULT 1, date TEXT, students INTEGER, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE)", "id, user_id, date, students"),
            ("waste_records", "CREATE TABLE waste_records (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER DEFAULT 1, date TEXT, prepared REAL, consumed REAL, waste REAL, rice_waste REAL DEFAULT 0.0, dal_waste REAL DEFAULT 0.0, vegetable_waste REAL DEFAULT 0.0, chapati_waste REAL DEFAULT 0.0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE)", "id, user_id, date, prepared, consumed, waste, rice_waste, dal_waste, vegetable_waste, chapati_waste, created_at"),
            ("inventory", "CREATE TABLE inventory (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER DEFAULT 1, item TEXT, quantity TEXT, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE)", "id, user_id, item, quantity")
        ]:
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{tbl}'")
            row = cursor.fetchone()
            if row and ("UNIQUE" in str(row[0])):
                temp_tbl = f"{tbl}_migration_temp"
                cursor.execute(f"CREATE TABLE {temp_tbl} AS SELECT * FROM {tbl}")
                cursor.execute(f"DROP TABLE {tbl}")
                cursor.execute(sql_create)
                cursor.execute(f"INSERT OR IGNORE INTO {tbl} SELECT * FROM {temp_tbl}")
                cursor.execute(f"DROP TABLE {temp_tbl}")

    except Exception as e:
        print(f"Migration notice: {e}")

def init_db():
    if USE_MYSQL:
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
        
        def run_ddl(sqlite_ddl, mysql_ddl):
            if USE_MYSQL:
                cursor.execute(mysql_ddl)
            else:
                cursor.execute(sqlite_ddl)
                
        # Users table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'user'
        )
        """, """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE,
            password VARCHAR(255),
            role VARCHAR(20) DEFAULT 'user'
        )
        """)
        
        # Predictions table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            date_str TEXT,
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
            festival TEXT,
            predicted_attendance INTEGER,
            rice_kg REAL,
            dal_kg REAL,
            curry_kg REAL,
            chapati_count INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """, """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT DEFAULT 1,
            date_str VARCHAR(20),
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
            festival VARCHAR(50),
            predicted_attendance INT,
            rice_kg FLOAT,
            dal_kg FLOAT,
            curry_kg FLOAT,
            chapati_count INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        
        migrate_sqlite_columns(cursor)
        
        # Feedback table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            username TEXT,
            feedback TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """, """
        CREATE TABLE IF NOT EXISTS feedback (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT DEFAULT 1,
            username VARCHAR(50),
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        
        # Attendance table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            date TEXT,
            students INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """, """
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT DEFAULT 1,
            date VARCHAR(20),
            students INT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        
        # Inventory table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            item TEXT,
            quantity TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """, """
        CREATE TABLE IF NOT EXISTS inventory (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT DEFAULT 1,
            item VARCHAR(50),
            quantity VARCHAR(30),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)

        # Donations table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS donations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            organization TEXT,
            food TEXT,
            quantity TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """, """
        CREATE TABLE IF NOT EXISTS donations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT DEFAULT 1,
            organization VARCHAR(100),
            food VARCHAR(100),
            quantity VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        
        # Expenses table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            date TEXT,
            amount REAL,
            description TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """, """
        CREATE TABLE IF NOT EXISTS expenses (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT DEFAULT 1,
            date VARCHAR(20),
            amount DOUBLE,
            description VARCHAR(255),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)

        # Waste records table
        run_ddl("""
        CREATE TABLE IF NOT EXISTS waste_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            date TEXT,
            prepared REAL,
            consumed REAL,
            waste REAL,
            rice_waste REAL DEFAULT 0,
            dal_waste REAL DEFAULT 0,
            vegetable_waste REAL DEFAULT 0,
            chapati_waste REAL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """, """
        CREATE TABLE IF NOT EXISTS waste_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT DEFAULT 1,
            date VARCHAR(20),
            prepared FLOAT,
            consumed FLOAT,
            waste FLOAT,
            rice_waste FLOAT DEFAULT 0,
            dal_waste FLOAT DEFAULT 0,
            vegetable_waste FLOAT DEFAULT 0,
            chapati_waste FLOAT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        
        conn.commit()
    
    admin_id = register_user("admin", "admin123", role="admin", allow_admin_creation=True)
    seed_default_inventory(admin_id or 1)
    seed_default_attendance_and_waste(admin_id or 1)

def register_user(username, password, role='user', allow_admin_creation=False):
    hashed = generate_password_hash(password)
    # Admin role is strictly forbidden for public registration unless explicitly allowed (e.g. system seed or admin portal)
    if (allow_admin_creation or username.lower() == 'admin') and role.lower() == 'admin':
        valid_role = 'admin'
    else:
        valid_role = 'user'

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            sql = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)" if USE_MYSQL else "INSERT INTO users (username, password, role) VALUES (?, ?, ?)"
            cursor.execute(sql, (username, hashed, valid_role))
            conn.commit()
            return cursor.lastrowid
    except Exception:
        # User might already exist, fetch id and ensure role is updated if admin
        with get_connection() as conn:
            cursor = conn.cursor()
            sql = "SELECT id, role FROM users WHERE username = %s" if USE_MYSQL else "SELECT id, role FROM users WHERE username = ?"
            cursor.execute(sql, (username,))
            row = cursor.fetchone()
            if row:
                u_id = row['id'] if USE_MYSQL else (row['id'] if isinstance(row, dict) else row[0])
                if username.lower() == 'admin':
                    up_sql = "UPDATE users SET role = 'admin' WHERE id = %s" if USE_MYSQL else "UPDATE users SET role = 'admin' WHERE id = ?"
                    cursor.execute(up_sql, (u_id,))
                    conn.commit()
                return u_id
            return None

def verify_user(username, password):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password, role FROM users WHERE username = %s" if USE_MYSQL else "SELECT id, username, password, role FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            stored_pwd = row['password'] if USE_MYSQL else (row['password'] if isinstance(row, dict) else row[2])
            user_id = row['id'] if USE_MYSQL else (row['id'] if isinstance(row, dict) else row[0])
            uname = row['username'] if USE_MYSQL else (row['username'] if isinstance(row, dict) else row[1])
            u_role = row['role'] if (USE_MYSQL or isinstance(row, dict)) else (row[3] if len(row) > 3 else 'user')
            if not u_role or uname.lower() == 'admin':
                u_role = 'admin' if uname.lower() == 'admin' else 'user'
            if check_password_hash(stored_pwd, password):
                return {
                    'id': user_id,
                    'username': uname,
                    'role': u_role,
                    'is_admin': (u_role == 'admin')
                }
def get_all_users():
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id, username, role, created_at FROM users ORDER BY id ASC")
        except Exception:
            cursor.execute("SELECT id, username, role FROM users ORDER BY id ASC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

def add_prediction(data_dict, user_id=1):
    sql = """
    INSERT INTO predictions (
        user_id, date_str, day, month, temperature, rainfall, humidity, wind_speed,
        breakfast_menu, lunch_menu, dinner_menu,
        previous_attendance, previous_waste, food_rating, festival, predicted_attendance,
        rice_kg, dal_kg, curry_kg, chapati_count
    ) VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
    """.format(p="%s" if USE_MYSQL else "?")
    
    values = (
        user_id, data_dict.get('Date', ''), data_dict['Day'], data_dict['Month'], data_dict['Temperature'], data_dict['Rainfall'],
        data_dict['Humidity'], data_dict['Wind_Speed'], data_dict['Breakfast_Menu'], data_dict['Lunch_Menu'],
        data_dict['Dinner_Menu'], data_dict['Previous_Attendance'], data_dict['Previous_Waste'],
        data_dict['Food_Rating'], data_dict.get('Festival', 'Normal Day'), data_dict['Predicted_Attendance'],
        data_dict.get('Rice_Kg', round(data_dict['Predicted_Attendance'] * 0.25, 1)),
        data_dict.get('Dal_Kg', round(data_dict['Predicted_Attendance'] * 0.12, 1)),
        data_dict.get('Curry_Kg', round(data_dict['Predicted_Attendance'] * 0.15, 1)),
        data_dict.get('Chapati_Count', int(data_dict['Predicted_Attendance'] * 2.5))
    )
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, values)
        conn.commit()

def get_predictions(search=None, user_id=None, is_admin=False):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        
        if is_admin:
            if search:
                q = f"SELECT p.*, u.username FROM predictions p LEFT JOIN users u ON p.user_id = u.id WHERE p.day LIKE {param} OR p.date_str LIKE {param} OR u.username LIKE {param} ORDER BY p.id DESC"
                cursor.execute(q, (f"%{search}%", f"%{search}%", f"%{search}%"))
            else:
                cursor.execute("SELECT p.*, u.username FROM predictions p LEFT JOIN users u ON p.user_id = u.id ORDER BY p.id DESC")
        else:
            if search:
                q = f"SELECT p.*, u.username FROM predictions p LEFT JOIN users u ON p.user_id = u.id WHERE p.user_id = {param} AND (p.day LIKE {param} OR p.date_str LIKE {param}) ORDER BY p.id DESC"
                cursor.execute(q, (user_id, f"%{search}%", f"%{search}%"))
            else:
                q = f"SELECT p.*, u.username FROM predictions p LEFT JOIN users u ON p.user_id = u.id WHERE p.user_id = {param} ORDER BY p.id DESC"
                cursor.execute(q, (user_id,))
                
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            results.append({
                'id': d.get('id'), 'user_id': d.get('user_id', 1), 'username': d.get('username', 'N/A'), 'date_str': d.get('date_str', 'N/A'),
                'day': d.get('day'), 'month': d.get('month'), 'temperature': d.get('temperature'),
                'rainfall': d.get('rainfall'), 'humidity': d.get('humidity'), 'wind_speed': d.get('wind_speed'),
                'breakfast_menu': d.get('breakfast_menu'), 'lunch_menu': d.get('lunch_menu'),
                'dinner_menu': d.get('dinner_menu'), 'previous_attendance': d.get('previous_attendance'),
                'previous_waste': d.get('previous_waste'), 'food_rating': d.get('food_rating'),
                'festival': d.get('festival', 'Normal Day'), 'predicted_attendance': d.get('predicted_attendance'),
                'rice_kg': d.get('rice_kg', round((d.get('predicted_attendance') or 750) * 0.25, 1)),
                'dal_kg': d.get('dal_kg', round((d.get('predicted_attendance') or 750) * 0.12, 1)),
                'curry_kg': d.get('curry_kg', round((d.get('predicted_attendance') or 750) * 0.15, 1)),
                'chapati_count': d.get('chapati_count', int((d.get('predicted_attendance') or 750) * 2.5)),
                'created_at': d.get('created_at', '')
            })
        return results

def delete_prediction(prediction_id, user_id=None, is_admin=False):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        if is_admin:
            cursor.execute(f"DELETE FROM predictions WHERE id = {param}", (prediction_id,))
        else:
            cursor.execute(f"DELETE FROM predictions WHERE id = {param} AND user_id = {param}", (prediction_id, user_id))
        conn.commit()
        return cursor.rowcount > 0

def add_feedback(username, text, user_id=1):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        cursor.execute(f"INSERT INTO feedback (user_id, username, feedback) VALUES ({param}, {param}, {param})", (user_id, username, text))
        conn.commit()

def get_feedback_count(user_id=None, is_admin=False):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        if is_admin:
            cursor.execute("SELECT COUNT(*) FROM feedback")
        else:
            cursor.execute(f"SELECT COUNT(*) FROM feedback WHERE user_id = {param}", (user_id,))
        row = cursor.fetchone()
        return row['COUNT(*)'] if USE_MYSQL else row[0]

def get_predictions_count(user_id=None, is_admin=False):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        if is_admin:
            cursor.execute("SELECT COUNT(*) FROM predictions")
        else:
            cursor.execute(f"SELECT COUNT(*) FROM predictions WHERE user_id = {param}", (user_id,))
        row = cursor.fetchone()
        return row['COUNT(*)'] if USE_MYSQL else row[0]

def get_users_count():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        row = cursor.fetchone()
        return row['COUNT(*)'] if USE_MYSQL else row[0]

def save_attendance(date, students, user_id=1):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        cursor.execute(f"INSERT INTO attendance (user_id, date, students) VALUES ({param}, {param}, {param})", (user_id, date, students))
        conn.commit()

def get_attendance(user_id=None, is_admin=False, limit=20):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        limit_val = int(limit)
        if is_admin:
            cursor.execute(f"SELECT a.id, a.user_id, a.date, a.students, u.username FROM attendance a LEFT JOIN users u ON a.user_id = u.id ORDER BY a.date DESC LIMIT {param}", (limit_val,))
        else:
            cursor.execute(f"SELECT id, user_id, date, students FROM attendance WHERE user_id = {param} ORDER BY date DESC LIMIT {param}", (user_id, limit_val))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d['students'] = int(d.get('students') or 0)
            except (ValueError, TypeError):
                d['students'] = 0
            results.append(d)
        return results

def seed_default_inventory(user_id=1):
    default_items = [
        (user_id, "Rice", "1500 Kg"),
        (user_id, "Dal", "500 Kg"),
        (user_id, "Vegetables", "800 Kg"),
        (user_id, "Cooking Oil", "250 L"),
        (user_id, "Spices", "150 Kg")
    ]
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inventory")
            cnt = cursor.fetchone()
            count_val = cnt['COUNT(*)'] if USE_MYSQL else cnt[0]
            if count_val == 0:
                stmt = "INSERT INTO inventory (user_id, item, quantity) VALUES (%s, %s, %s)" if USE_MYSQL else "INSERT INTO inventory (user_id, item, quantity) VALUES (?, ?, ?)"
                cursor.executemany(stmt, default_items)
                conn.commit()
    except Exception:
        pass

def seed_default_attendance_and_waste(user_id=1):
    default_att = [
        (user_id, "2026-08-01", 720),
        (user_id, "2026-08-02", 740),
        (user_id, "2026-08-03", 680),
        (user_id, "2026-08-04", 760),
        (user_id, "2026-08-05", 790),
        (user_id, "2026-08-06", 810),
        (user_id, "2026-08-07", 750)
    ]
    default_waste = [
        (user_id, "2026-08-01", 380.0, 365.0, 15.0),
        (user_id, "2026-08-02", 400.0, 382.0, 18.0),
        (user_id, "2026-08-03", 360.0, 348.0, 12.0),
        (user_id, "2026-08-04", 410.0, 396.0, 14.0),
        (user_id, "2026-08-05", 420.0, 404.0, 16.0),
        (user_id, "2026-08-06", 430.0, 417.0, 13.0),
        (user_id, "2026-08-07", 400.0, 389.0, 11.0)
    ]
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM attendance")
            cnt_att = cursor.fetchone()
            count_att = cnt_att['COUNT(*)'] if USE_MYSQL else cnt_att[0]
            if count_att == 0:
                stmt_att = "INSERT INTO attendance (user_id, date, students) VALUES (%s, %s, %s)" if USE_MYSQL else "INSERT INTO attendance (user_id, date, students) VALUES (?, ?, ?)"
                cursor.executemany(stmt_att, default_att)
                
            cursor.execute("SELECT COUNT(*) FROM waste_records")
            cnt_w = cursor.fetchone()
            count_w = cnt_w['COUNT(*)'] if USE_MYSQL else cnt_w[0]
            if count_w == 0:
                stmt_w = "INSERT INTO waste_records (user_id, date, prepared, consumed, waste) VALUES (%s, %s, %s, %s, %s)" if USE_MYSQL else "INSERT INTO waste_records (user_id, date, prepared, consumed, waste) VALUES (?, ?, ?, ?, ?)"
                cursor.executemany(stmt_w, default_waste)
                
            conn.commit()
    except Exception:
        pass

def save_inventory(item, quantity, user_id=1):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        cursor.execute(f"INSERT INTO inventory (user_id, item, quantity) VALUES ({param}, {param}, {param})", (user_id, item, quantity))
        conn.commit()

def get_inventory(user_id=None, is_admin=False):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        if is_admin:
            cursor.execute("SELECT i.id, i.user_id, i.item, i.quantity, u.username FROM inventory i LEFT JOIN users u ON i.user_id = u.id ORDER BY i.id DESC")
        else:
            cursor.execute(f"SELECT id, user_id, item, quantity FROM inventory WHERE user_id = {param} ORDER BY id DESC", (user_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

def save_expense(date, amount, description, user_id=1):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        cursor.execute(f"INSERT INTO expenses (user_id, date, amount, description) VALUES ({param}, {param}, {param}, {param})", (user_id, date, amount, description))
        conn.commit()

def get_expense_stats(user_id=None, is_admin=False):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        if is_admin:
            cursor.execute("SELECT SUM(amount) FROM expenses")
        else:
            cursor.execute(f"SELECT SUM(amount) FROM expenses WHERE user_id = {param}", (user_id,))
        row = cursor.fetchone()
        val = (row['SUM(amount)'] if USE_MYSQL else row[0]) if row else 0.0
        total_expense = float(val) if val is not None else 0.0
        monthly_savings = max(15000.0, 120000.0 - total_expense)
        return total_expense, monthly_savings

def save_waste_record(date, prepared, consumed, waste, rice_waste=0.0, dal_waste=0.0, vegetable_waste=0.0, chapati_waste=0.0, user_id=1):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        try:
            cursor.execute(
                f"INSERT INTO waste_records (user_id, date, prepared, consumed, waste, rice_waste, dal_waste, vegetable_waste, chapati_waste) VALUES ({param}, {param}, {param}, {param}, {param}, {param}, {param}, {param}, {param})",
                (user_id, date, float(prepared), float(consumed), float(waste), float(rice_waste), float(dal_waste), float(vegetable_waste), float(chapati_waste))
            )
        except Exception:
            cursor.execute(
                f"INSERT INTO waste_records (user_id, date, prepared, consumed, waste) VALUES ({param}, {param}, {param}, {param}, {param})",
                (user_id, date, float(prepared), float(consumed), float(waste))
            )
        conn.commit()

def get_waste_records(user_id=None, is_admin=False, limit=15):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        limit_val = int(limit)
        try:
            if is_admin:
                cursor.execute(f"SELECT w.id, w.user_id, w.date, w.prepared, w.consumed, w.waste, w.rice_waste, w.dal_waste, w.vegetable_waste, w.chapati_waste, u.username FROM waste_records w LEFT JOIN users u ON w.user_id = u.id ORDER BY w.date DESC LIMIT {param}", (limit_val,))
            else:
                cursor.execute(f"SELECT id, user_id, date, prepared, consumed, waste, rice_waste, dal_waste, vegetable_waste, chapati_waste FROM waste_records WHERE user_id = {param} ORDER BY date DESC LIMIT {param}", (user_id, limit_val))
        except Exception:
            if is_admin:
                cursor.execute(f"SELECT w.id, w.user_id, w.date, w.prepared, w.consumed, w.waste, u.username FROM waste_records w LEFT JOIN users u ON w.user_id = u.id ORDER BY w.date DESC LIMIT {param}", (limit_val,))
            else:
                cursor.execute(f"SELECT id, user_id, date, prepared, consumed, waste FROM waste_records WHERE user_id = {param} ORDER BY date DESC LIMIT {param}", (user_id, limit_val))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                prep = float(d.get('prepared') or 0.0)
            except (ValueError, TypeError):
                prep = 0.0
            try:
                cons = float(d.get('consumed') or 0.0)
            except (ValueError, TypeError):
                cons = 0.0
            try:
                wst = float(d.get('waste') or 0.0)
            except (ValueError, TypeError):
                wst = 0.0
            d['prepared'] = prep
            d['consumed'] = cons
            d['waste'] = wst
            d['rice_waste'] = float(d.get('rice_waste') or 0.0)
            d['dal_waste'] = float(d.get('dal_waste') or 0.0)
            d['vegetable_waste'] = float(d.get('vegetable_waste') or 0.0)
            d['chapati_waste'] = float(d.get('chapati_waste') or 0.0)
            results.append(d)
        return results

def get_total_food_waste(user_id=None, is_admin=False):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        if is_admin:
            cursor.execute("SELECT SUM(waste) FROM waste_records")
        else:
            cursor.execute(f"SELECT SUM(waste) FROM waste_records WHERE user_id = {param}", (user_id,))
        row = cursor.fetchone()
        val = (row['SUM(waste)'] if USE_MYSQL else row[0]) if row else 0.0
        return round(float(val), 1) if val is not None else 0.0

def save_donation(organization, food, quantity, user_id=1):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        cursor.execute(f"INSERT INTO donations (user_id, organization, food, quantity) VALUES ({param}, {param}, {param}, {param})", (user_id, organization, food, quantity))
        conn.commit()

def get_donations(user_id=None, is_admin=False):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        if is_admin:
            cursor.execute("SELECT d.id, d.user_id, d.organization, d.food, d.quantity, d.created_at, u.username FROM donations d LEFT JOIN users u ON d.user_id = u.id ORDER BY d.id DESC")
        else:
            cursor.execute(f"SELECT id, user_id, organization, food, quantity, created_at FROM donations WHERE user_id = {param} ORDER BY id DESC", (user_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

def get_feedback(user_id=None, is_admin=False):
    with get_connection() as conn:
        cursor = conn.cursor()
        param = "%s" if USE_MYSQL else "?"
        if is_admin:
            cursor.execute("SELECT f.id, f.user_id, f.username, f.feedback, f.created_at FROM feedback f ORDER BY f.id DESC")
        else:
            cursor.execute(f"SELECT id, user_id, username, feedback, created_at FROM feedback WHERE user_id = {param} ORDER BY id DESC", (user_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

def get_recent_activities(limit=5):
    """Returns recent activities across users for Admin dashboard."""
    activities = []
    with get_connection() as conn:
        cursor = conn.cursor()
        # Predictions
        cursor.execute("SELECT p.created_at, u.username, 'Prediction' as activity_type, CONCAT('Generated prediction for ', p.predicted_attendance, ' students') as details FROM predictions p LEFT JOIN users u ON p.user_id = u.id ORDER BY p.id DESC LIMIT 5" if USE_MYSQL else "SELECT p.created_at, u.username, 'Prediction' as activity_type, ('Generated prediction for ' || p.predicted_attendance || ' students') as details FROM predictions p LEFT JOIN users u ON p.user_id = u.id ORDER BY p.id DESC LIMIT 5")
        rows = cursor.fetchall()
        for r in rows:
            d = dict(r)
            activities.append({
                'created_at': d.get('created_at', ''),
                'username': d.get('username', 'System'),
                'type': d.get('activity_type', 'Prediction'),
                'details': d.get('details', '')
            })
    # Sort combined activities by created_at descending
    activities.sort(key=lambda x: str(x['created_at']), reverse=True)
    return activities[:limit]

