from flask import Flask, render_template, request, send_file, redirect, session, flash, url_for, jsonify
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os
import secrets
import datetime
import database
from werkzeug.utils import secure_filename
from model.prediction import predict_attendance, predict_and_recommend, get_model_metrics

app = Flask(__name__)
# Secret key configured via environment variable with documented development fallback
app.secret_key = os.environ.get("SECRET_KEY", "btech-hostel-reduction-placement-secret-key-dev-fallback")
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max file upload size

# CSRF Protection Helpers
def get_csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(16)
    return session['csrf_token']

app.jinja_env.globals['csrf_token'] = get_csrf_token

@app.before_request
def csrf_protect():
    if request.method == 'POST':
        token = session.get('csrf_token')
        form_token = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token')
        if not token or form_token != token:
            flash("Security Warning: CSRF token missing or invalid. Please try submitting again.")
            return redirect(request.referrer or url_for('home'))

# Initialize Database
database.init_db()

def get_current_user():
    user_id = session.get('user_id', 1)
    role = session.get('role', 'user')
    is_admin = (role == 'admin')
    return user_id, role, is_admin

# Decorator to secure application routes
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            flash("Authorization required to access this system.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            flash("Authorization required to access this system.")
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash("Admin privileges required to access this feature.")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def parse_date_info(date_str=None):
    if not date_str:
        dt = datetime.date.today()
    else:
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            dt = datetime.date.today()
            
    days_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months_map = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    
    day_name = days_map[dt.weekday()]
    month_name = months_map[dt.month - 1]
    
    # Derive Season
    if month_name in ["December", "January", "February"]:
        season = "Winter"
    elif month_name in ["March", "April", "May"]:
        season = "Summer"
    elif month_name in ["June", "July", "August", "September"]:
        season = "Monsoon"
    else:
        season = "Autumn"
        
    return {
        "date_str": dt.strftime("%Y-%m-%d"),
        "day": day_name,
        "month": month_name,
        "season": season,
        "prev_date": (dt - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        "next_date": (dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        "today_date": datetime.date.today().strftime("%Y-%m-%d")
    }

# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
@login_required
def dashboard():
    user_id, role, is_admin = get_current_user()
    
    # User-specific or Admin aggregate statistics
    today_attendance = 750
    predicted_attendance = 780
    predicted_rice = round(780 * 0.25, 1)
    today_waste = 14.2
    
    att_list = database.get_attendance(user_id=user_id, is_admin=is_admin, limit=1)
    if att_list:
        today_attendance = att_list[0]['students']
        
    pred_list = database.get_predictions(user_id=user_id, is_admin=is_admin)
    if pred_list:
        predicted_attendance = pred_list[0]['predicted_attendance']
        predicted_rice = pred_list[0]['rice_kg']
        
    waste_list = database.get_waste_records(user_id=user_id, is_admin=is_admin, limit=1)
    if waste_list:
        today_waste = waste_list[0]['waste']

    # Role-based dashboard indicators
    admin_metrics = {}
    user_metrics = {}

    if is_admin:
        admin_metrics = {
            'total_users': database.get_users_count(),
            'total_predictions': database.get_predictions_count(is_admin=True),
            'overall_food_waste': database.get_total_food_waste(is_admin=True),
            'overall_cost_savings': database.get_expense_stats(is_admin=True)[1],
            'recent_activities': database.get_recent_activities(limit=5)
        }
    else:
        user_metrics = {
            'my_predictions_count': database.get_predictions_count(user_id=user_id, is_admin=False),
            'my_attendance': today_attendance,
            'my_reports_count': database.get_predictions_count(user_id=user_id, is_admin=False),
            'my_food_waste_history': database.get_total_food_waste(user_id=user_id, is_admin=False),
            'my_savings': database.get_expense_stats(user_id=user_id, is_admin=False)[1]
        }
    
    # Custom Menu Check
    menu_rows = []
    menu_image = None
    menu_pdf = None
    menu_doc = None
    
    upload_dir = os.path.join("static", "uploads")
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            if f.startswith("uploaded_menu."):
                ext = f.split('.')[-1].lower()
                rel_path = f"uploads/{f}"
                if ext in ['png', 'jpg', 'jpeg']:
                    menu_image = rel_path
                elif ext == 'pdf':
                    menu_pdf = rel_path
                elif ext in ['doc', 'docx']:
                    menu_doc = rel_path
                    
    if not menu_image and not menu_pdf and not menu_doc:
        menu_path = os.path.join("dataset", "hostel_menu.csv")
        if os.path.exists(menu_path):
            try:
                menu_df = pd.read_csv(menu_path)
                menu_rows = menu_df.to_dict(orient='records')
            except Exception:
                pass
                
    model_metrics = get_model_metrics()
    best_model_info = model_metrics[0] if model_metrics else {"Model": "Random Forest", "R2": 0.912, "MAE": 16.75, "RMSE": 20.74}
    
    return render_template(
        'dashboard.html',
        is_admin=is_admin,
        role=role,
        admin_metrics=admin_metrics,
        user_metrics=user_metrics,
        today_attendance=today_attendance,
        predicted_attendance=predicted_attendance,
        predicted_rice=predicted_rice,
        today_waste=today_waste,
        menu_list=menu_rows,
        menu_image=menu_image,
        menu_pdf=menu_pdf,
        menu_doc=menu_doc,
        best_model=best_model_info
    )

# ---------------- PREDICTION & RECOMMENDATION ----------------
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    user_id, role, is_admin = get_current_user()
    selected_date = request.args.get('date', datetime.date.today().strftime("%Y-%m-%d"))
    date_info = parse_date_info(selected_date)
    
    prediction = None
    recommendation = None
    
    if request.method == 'POST':
        try:
            req_date = request.form.get('date', date_info['date_str'])
            date_info = parse_date_info(req_date)
            
            season_input = request.form.get('Season', date_info['season'])
            festival_input = request.form.get('Festival', 'Normal Day')
            
            temp = float(request.form['Temperature'])
            rainfall = float(request.form['Rainfall'])
            humidity = int(request.form['Humidity'])
            wind_speed = float(request.form['Wind_Speed'])
            prev_att = int(request.form['Previous_Attendance'])
            prev_waste = float(request.form['Previous_Waste'])
            food_rating = float(request.form['Food_Rating'])
            
            # Input validation range checks
            if not (-10.0 <= temp <= 60.0):
                raise ValueError("Temperature must be between -10°C and 60°C.")
            if not (0.0 <= rainfall <= 1000.0):
                raise ValueError("Rainfall must be between 0 and 1000 mm.")
            if not (0 <= humidity <= 100):
                raise ValueError("Humidity must be between 0% and 100%.")
            if not (0.0 <= wind_speed <= 200.0):
                raise ValueError("Wind Speed must be between 0 and 200 km/h.")
            if not (1 <= prev_att <= 2000):
                raise ValueError("Previous Attendance must be between 1 and 2000 students.")
            if not (0.0 <= prev_waste <= 500.0):
                raise ValueError("Previous Waste must be between 0 and 500 kg.")
            if not (1.0 <= food_rating <= 5.0):
                raise ValueError("Food Rating must be between 1.0 and 5.0.")

            input_features = {
                'Date': date_info['date_str'],
                'Day': request.form.get('Day', date_info['day']),
                'Month': request.form.get('Month', date_info['month']),
                'Season': season_input,
                'Temperature': temp,
                'Rainfall': rainfall,
                'Humidity': humidity,
                'Wind_Speed': wind_speed,
                'Breakfast_Menu': request.form.get('Breakfast_Menu', 'Idli / Dosa'),
                'Lunch_Menu': request.form.get('Lunch_Menu', 'Rice & Dal'),
                'Dinner_Menu': request.form.get('Dinner_Menu', 'Roti & Sabzi'),
                'Previous_Attendance': prev_att,
                'Previous_Waste': prev_waste,
                'Food_Rating': food_rating,
                'Festival': festival_input
            }
            
            pred_attendance, rec_dict = predict_and_recommend(input_features, festival=festival_input)
            
            input_features['Predicted_Attendance'] = pred_attendance
            input_features['Rice_Kg'] = rec_dict['rice_kg']
            input_features['Dal_Kg'] = rec_dict['dal_kg']
            input_features['Curry_Kg'] = rec_dict['curry_kg']
            input_features['Chapati_Count'] = rec_dict['chapati_count']
            
            database.add_prediction(input_features, user_id=user_id)
            
            prediction = {
                'students': pred_attendance,
                'rice': rec_dict['rice_kg'],
                'dal': rec_dict['dal_kg'],
                'curry': rec_dict['curry_kg'],
                'chapati': rec_dict['chapati_count']
            }
            recommendation = rec_dict
            flash(f"ML Forecasting & Food Recommendation completed successfully for {date_info['date_str']}!")
        except Exception as e:
            flash(f"Inference pipeline execution error: {str(e)}")
            
    return render_template(
        'prediction.html',
        prediction=prediction,
        recommendation=recommendation,
        date_info=date_info
    )

# ---------------- ATTENDANCE ----------------
@app.route('/attendance', methods=['GET', 'POST'])
@login_required
def attendance():
    user_id, role, is_admin = get_current_user()
    selected_date = request.args.get('date', datetime.date.today().strftime("%Y-%m-%d"))
    date_info = parse_date_info(selected_date)
    
    if request.method == 'POST':
        try:
            date_val = request.form['date']
            students_val = int(request.form['students'])
            if not (1 <= students_val <= 2000):
                raise ValueError("Attendance must be between 1 and 2000 students.")
            database.save_attendance(date_val, students_val, user_id=user_id)
            flash(f"Attendance registered successfully: {students_val} students on {date_val}.")
            date_info = parse_date_info(date_val)
        except Exception as e:
            flash(f"Error registering attendance: {str(e)}")
            
    attendance_list = database.get_attendance(user_id=user_id, is_admin=is_admin)
    return render_template('attendance.html', attendance_list=attendance_list, date_info=date_info)

# ---------------- WASTE ANALYSIS ----------------
@app.route('/waste', methods=['GET', 'POST'])
@login_required
def waste():
    user_id, role, is_admin = get_current_user()
    selected_date = request.args.get('date', datetime.date.today().strftime("%Y-%m-%d"))
    date_info = parse_date_info(selected_date)
    
    if request.method == 'POST':
        try:
            date_val = request.form['date']
            prep = float(request.form['prepared'])
            cons = float(request.form['consumed'])
            
            if prep < 0 or cons < 0:
                flash("Error: Quantities cannot be negative.")
            elif prep < cons:
                flash("Error: Consumed quantity cannot exceed prepared volume.")
            else:
                w_val = round(prep - cons, 2)
                database.save_waste_record(date_val, prep, cons, w_val, user_id=user_id)
                flash(f"Wastage recorded: {w_val} Kg of food for {date_val}.")
                date_info = parse_date_info(date_val)
        except Exception as e:
            flash(f"Error logging waste: {str(e)}")
            
    waste_list = database.get_waste_records(user_id=user_id, is_admin=is_admin)
    return render_template('waste.html', waste_list=waste_list, date_info=date_info)

# ---------------- REPORTS ----------------
@app.route('/reports')
@login_required
def reports():
    user_id, role, is_admin = get_current_user()
    total_preds = database.get_predictions_count(user_id=user_id, is_admin=is_admin)
    total_expense, savings = database.get_expense_stats(user_id=user_id, is_admin=is_admin)
    
    conn = database.get_connection()
    cursor = conn.cursor()
    param = "%s" if database.USE_MYSQL else "?"
    if is_admin:
        cursor.execute("SELECT AVG(predicted_attendance) FROM predictions")
    else:
        cursor.execute(f"SELECT AVG(predicted_attendance) FROM predictions WHERE user_id = {param}", (user_id,))
    avg_row = cursor.fetchone()
    avg_students = round(avg_row['AVG(predicted_attendance)'] if database.USE_MYSQL else (avg_row[0] or 760), 1) if avg_row else 760.0
    conn.close()
    
    model_metrics = get_model_metrics()
    
    return render_template(
        'reports.html',
        total_predictions=total_preds,
        avg_students=avg_students,
        total_expense=total_expense,
        savings=savings,
        model_metrics=model_metrics
    )

# ---------------- SETTINGS ----------------
@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html', use_mysql=database.USE_MYSQL)

# ---------------- UPLOAD MENU ----------------
@app.route('/upload_menu', methods=['POST'])
@login_required
def upload_menu():
    if 'menu_file' not in request.files:
        flash("No file part in the upload request.")
        return redirect(url_for('settings'))
    file = request.files['menu_file']
    if file.filename == '':
        flash("No file selected for upload.")
        return redirect(url_for('settings'))
        
    allowed_extensions = {'csv', 'png', 'jpg', 'jpeg', 'pdf', 'doc', 'docx'}
    filename_raw = secure_filename(file.filename)
    if not filename_raw or '.' not in filename_raw:
        flash("Invalid file name format.")
        return redirect(url_for('settings'))
        
    ext = filename_raw.rsplit('.', 1)[1].lower()
    
    if ext in allowed_extensions:
        try:
            upload_dir = os.path.join("static", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            for f in os.listdir(upload_dir):
                if f.startswith("uploaded_menu."):
                    try:
                        os.remove(os.path.join(upload_dir, f))
                    except Exception:
                        pass
                
            safe_name = f"uploaded_menu.{ext}"
            file.save(os.path.join(upload_dir, safe_name))
            flash(f"Weekly hostel menu uploaded successfully ({ext.upper()} format).")
        except Exception as e:
            flash(f"Error saving menu file: {str(e)}")
    else:
        flash("Invalid file format. Allowed: CSV, Images (PNG/JPG), PDF, DOC, DOCX.")
        
    return redirect(url_for('settings'))

# ---------------- REGISTER ----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            flash("Username and password are required.")
            return render_template('register.html')

        # Public registration strictly forces role='user' regardless of request parameters
        if database.register_user(username, password, role='user', allow_admin_creation=False):
            flash("User registration successful! Please log in.")
            return redirect(url_for('login'))
        else:
            flash("Username already exists or registration failed.")
    return render_template('register.html')

# ---------------- INVENTORY ----------------
@app.route('/inventory', methods=['GET', 'POST'])
@login_required
def inventory():
    user_id, role, is_admin = get_current_user()
    if request.method == 'POST':
        item = request.form['item']
        quantity = request.form['quantity']
        database.save_inventory(item, quantity, user_id=user_id)
        flash(f"Inventory saved: {item} updated to {quantity}.")
    items = [(row['item'], row['quantity']) for row in database.get_inventory(user_id=user_id, is_admin=is_admin)]
    return render_template('inventory.html', items=items)

# ---------------- EXPENSE ----------------
@app.route('/expense', methods=['GET', 'POST'])
@login_required
def expense():
    user_id, role, is_admin = get_current_user()
    selected_date = request.args.get('date', datetime.date.today().strftime("%Y-%m-%d"))
    date_info = parse_date_info(selected_date)
    
    if request.method == 'POST':
        date_val = request.form['date']
        try:
            amount = float(request.form['amount'])
            desc = request.form['description']
            database.save_expense(date_val, amount, desc, user_id=user_id)
            flash("Expense recorded successfully.")
            date_info = parse_date_info(date_val)
        except ValueError:
            flash("Invalid expense amount.")
            
    total_expense, savings = database.get_expense_stats(user_id=user_id, is_admin=is_admin)
    return render_template('expense.html', expense=total_expense, savings=savings, date_info=date_info)

# ---------------- FOOD DONATION ----------------
@app.route('/donation', methods=['GET', 'POST'])
@login_required
def donation():
    user_id, role, is_admin = get_current_user()
    if request.method == 'POST':
        try:
            org = request.form['organization']
            food_item = request.form['food']
            qty = request.form['quantity']
            database.save_donation(org, food_item, qty, user_id=user_id)
            flash("Surplus food donation registered successfully!")
        except Exception as e:
            flash(f"Error registering donation: {str(e)}")
            
    donations_list = database.get_donations(user_id=user_id, is_admin=is_admin)
    return render_template('donation.html', donations=donations_list)

# ---------------- FEEDBACK ----------------
@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    user_id, role, is_admin = get_current_user()
    if request.method == 'POST':
        username = session.get('user', request.form.get('username', 'Anonymous'))
        text = request.form['feedback']
        database.add_feedback(username, text, user_id=user_id)
        flash("Thank you! Feedback submitted successfully.")
        return redirect(url_for('dashboard'))
    return render_template('feedback.html')

# ---------------- HISTORY ----------------
@app.route('/history', methods=['GET', 'POST'])
@login_required
def history():
    user_id, role, is_admin = get_current_user()
    search = None
    if request.method == 'POST':
        search = request.form.get('search')
    
    predictions = database.get_predictions(search, user_id=user_id, is_admin=is_admin)
    rows = []
    for p in predictions:
        students = p['predicted_attendance']
        rows.append({
            'id': p['id'],
            'user_id': p.get('user_id'),
            'username': p.get('username', 'N/A'),
            'date_str': p.get('date_str', 'N/A'),
            'day': p['day'],
            'month': p['month'],
            'students': students,
            'rice': p.get('rice_kg', round(students * 0.25, 1)),
            'dal': p.get('dal_kg', round(students * 0.12, 1)),
            'curry': p.get('curry_kg', round(students * 0.15, 1)),
            'chapati': p.get('chapati_count', int(students * 2.5)),
            'festival': p.get('festival', 'Normal Day'),
            'created_at': p.get('created_at', '')
        })
    return render_template('history.html', rows=rows, search=search, is_admin=is_admin)

# ---------------- DELETE HISTORY ----------------
@app.route('/delete/<int:prediction_id>')
@login_required
def delete_prediction(prediction_id):
    user_id, role, is_admin = get_current_user()
    success = database.delete_prediction(prediction_id, user_id=user_id, is_admin=is_admin)
    if success:
        flash("Forecast record deleted successfully.")
    else:
        flash("Unauthorized or record not found: You can only delete your own prediction entries.")
    return redirect(url_for('history'))

# ---------------- NOTIFICATIONS ----------------
@app.route('/notifications')
@login_required
def notifications():
    return render_template('notifications.html')

# ---------------- ABOUT ----------------
@app.route('/about')
def about():
    return render_template('about.html')

# ---------------- CONTACT ----------------
@app.route('/contact')
def contact():
    return render_template('contact.html')

# ---------------- SIMPLE PREDICT ----------------
@app.route('/simple_predict', methods=['GET', 'POST'])
@login_required
def simple_predict():
    prediction = None
    if request.method == 'POST':
        try:
            students = int(request.form['students'])
            prediction = {
                'rice': round(students * 0.25, 1),
                'chapati': int(students * 2.5),
                'curry': round(students * 0.15, 1),
                'dal': round(students * 0.12, 1)
            }
        except ValueError:
            flash("Invalid number of students.")
    return render_template('predict.html', prediction=prediction)

# ---------------- ML FORECAST ----------------
@app.route('/ml_forecast')
@login_required
def ml_forecast():
    user_id, role, is_admin = get_current_user()
    predictions = database.get_predictions(user_id=user_id, is_admin=is_admin)
    
    if predictions:
        latest = predictions[0]
        predicted_students = latest['predicted_attendance']
        rice = latest['rice_kg']
        dal = latest['dal_kg']
        curry = latest['curry_kg']
        chapati = latest['chapati_count']
    else:
        predicted_students = 780
        rice = round(780 * 0.25, 1)
        dal = round(780 * 0.12, 1)
        curry = round(780 * 0.15, 1)
        chapati = int(780 * 2.5)
        
    model_metrics = get_model_metrics()
    
    return render_template(
        'ml_predict.html',
        predicted_students=predicted_students,
        rice=rice,
        curry=curry,
        chapati=chapati,
        dal=dal,
        model_metrics=model_metrics
    )

# ---------------- ADMIN PANEL ----------------
@app.route('/admin')
@admin_required
def admin():
    users_count = database.get_users_count()
    users_list = database.get_all_users()
    feedback_count = database.get_feedback_count(is_admin=True)
    predictions_count = database.get_predictions_count(is_admin=True)
    feedback_list = database.get_feedback(is_admin=True)
    return render_template(
        'admin.html',
        users=users_count,
        users_list=users_list,
        feedback=feedback_count,
        predictions=predictions_count,
        feedback_list=feedback_list
    )

# ---------------- LOGIN & LOGOUT ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user_info = database.verify_user(username, password)
        if user_info:
            session['user_id'] = user_info['id']
            session['user'] = user_info['username']
            session['role'] = user_info['role']
            session['is_admin'] = user_info['is_admin']
            flash(f"Welcome back {user_info['username']} ({user_info['role'].capitalize()}). Authentication successful!")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user', None)
    session.pop('role', None)
    session.pop('is_admin', None)
    flash("You have been signed out of the system.")
    return redirect(url_for('home'))

# ---------------- API MULTI-DATASET CHART ENDPOINTS ----------------
@app.route('/api/dashboard_chart_data')
@login_required
def dashboard_chart_data():
    user_id, role, is_admin = get_current_user()
    conn = database.get_connection()
    cursor = conn.cursor()
    param = "%s" if database.USE_MYSQL else "?"
    
    # 1. Attendance logs from DB for user or admin
    if is_admin:
        cursor.execute("SELECT date, students FROM attendance ORDER BY date ASC LIMIT 10")
    else:
        cursor.execute(f"SELECT date, students FROM attendance WHERE user_id = {param} ORDER BY date ASC LIMIT 10", (user_id,))
    att_rows = cursor.fetchall()
    
    att_labels = []
    att_values = []
    for r in att_rows:
        att_labels.append(r['date'] if database.USE_MYSQL else r[0])
        att_values.append(r['students'] if database.USE_MYSQL else r[1])
        
    # 2. Waste logs from DB for user or admin
    if is_admin:
        cursor.execute("SELECT date, prepared, consumed, waste FROM waste_records ORDER BY date ASC LIMIT 10")
    else:
        cursor.execute(f"SELECT date, prepared, consumed, waste FROM waste_records WHERE user_id = {param} ORDER BY date ASC LIMIT 10", (user_id,))
    w_rows = cursor.fetchall()
    
    w_labels = []
    w_values = []
    prep_values = []
    consumed_values = []
    for r in w_rows:
        w_labels.append(r['date'] if database.USE_MYSQL else r[0])
        prep_values.append(r['prepared'] if database.USE_MYSQL else r[1])
        consumed_values.append(r['consumed'] if database.USE_MYSQL else r[2])
        w_values.append(r['waste'] if database.USE_MYSQL else r[3])
        
    if not att_labels:
        att_labels = ["Aug 01", "Aug 02", "Aug 03", "Aug 04", "Aug 05", "Aug 06", "Aug 07"]
        att_values = [720, 740, 680, 760, 790, 810, 750]
    if not w_labels:
        w_labels = ["Aug 01", "Aug 02", "Aug 03", "Aug 04", "Aug 05", "Aug 06", "Aug 07"]
        prep_values = [380, 400, 360, 410, 420, 430, 400]
        consumed_values = [365, 382, 348, 396, 404, 417, 389]
        w_values = [15, 18, 12, 14, 16, 13, 11]
        
    total_expense, savings = database.get_expense_stats(user_id=user_id, is_admin=is_admin)
    conn.close()
    
    # 3. Multi-Dataset 2: Catering & Event Waste (`food_wastage_data.csv`)
    catering_waste = {"Meat": 32.5, "Vegetables": 22.1, "Rice": 18.4, "Bread": 15.0, "Fish": 28.0}
    if os.path.exists("food_wastage_data.csv"):
        try:
            df_fw = pd.read_csv("food_wastage_data.csv").drop_duplicates()
            grp = df_fw.groupby("Type of Food")["Wastage Food Amount"].mean().round(1)
            catering_waste = grp.to_dict()
        except Exception:
            pass
            
    # 4. Multi-Dataset 3: Canteen Meal Waste & Cost Losses (`Dataset Propely.csv`)
    canteen_losses = {"Breakfast": 320.0, "Lunch": 540.0, "Dinner": 410.0}
    canteen_sections = {"Section A": 410.0, "Section B": 380.0, "Section C": 450.0, "Section D": 320.0}
    if os.path.exists("Dataset Propely.csv"):
        try:
            df_prop = pd.read_csv("Dataset Propely.csv")
            meal_grp = df_prop.groupby("Meal")["Cost_Loss"].sum().round(1)
            sec_grp = df_prop.groupby("Canteen_Section")["Cost_Loss"].sum().round(1)
            canteen_losses = meal_grp.to_dict()
            canteen_sections = {f"Section {k}": v for k, v in sec_grp.to_dict().items()}
        except Exception:
            pass

    # 5. Multi-Dataset 4: Global Country Waste Benchmarks (`global_food_wastage_dataset.csv`)
    global_benchmarks = {"India": 55.4, "USA": 88.2, "China": 64.1, "Germany": 72.3, "Japan": 59.8}
    if os.path.exists("global_food_wastage_dataset.csv"):
        try:
            df_glob = pd.read_csv("global_food_wastage_dataset.csv")
            c_grp = df_glob.groupby("Country")["Avg Waste per Capita (Kg)"].mean().head(6).round(1)
            global_benchmarks = c_grp.to_dict()
        except Exception:
            pass

    # 6. ML Model Performance Comparison
    model_metrics = get_model_metrics()
    
    return jsonify({
        'attendance_labels': att_labels,
        'attendance_values': att_values,
        'waste_labels': w_labels,
        'waste_values': w_values,
        'prep_values': prep_values,
        'consumed_values': consumed_values,
        'total_expense': total_expense,
        'savings': savings,
        'catering_waste': catering_waste,
        'canteen_losses': canteen_losses,
        'canteen_sections': canteen_sections,
        'global_benchmarks': global_benchmarks,
        'model_metrics': model_metrics
    })

# ---------------- DOWNLOAD PDF REPORT ----------------
@app.route('/download_report')
@login_required
def download_report():
    user_id, role, is_admin = get_current_user()
    report_path = "Prediction_Report.pdf"
    
    total_preds = database.get_predictions_count(user_id=user_id, is_admin=is_admin)
    total_expense, savings = database.get_expense_stats(user_id=user_id, is_admin=is_admin)
    
    doc = SimpleDocTemplate(report_path, pagesize=letter,
                            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#00f2fe'),
        spaceAfter=15
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        textColor=colors.HexColor('#94a3b8'),
        spaceAfter=30
    )
    
    story.append(Paragraph("Smart Hostel AI Food Management System", title_style))
    story.append(Paragraph("Major Project Audit & Waste Telemetry Report (Capacity: 500-1000 Students)", subtitle_style))
    
    data = [
        ["Key Telemetry Metric", "Calculated Value"],
        ["User Account Context", f"ID {user_id} ({role.capitalize()})"],
        ["Hostel Registered Capacity", "500 - 1000 Students"],
        ["Forecast Runs Executed", str(total_preds)],
        ["Recorded Daily Food Waste (Average)", "14.2 Kg"],
        ["Total Logged Expenses", f"Rs. {total_expense:.2f}"],
        ["Net Monthly Cost Savings Balance", f"Rs. {savings:.2f}"]
    ]
    
    t = Table(data, colWidths=[240, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#131a26')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
    ]))
    
    story.append(t)
    doc.build(story)
    
    return send_file(report_path, as_attachment=True)

# ---------------- EXPORT EXCEL ----------------
@app.route('/export_excel')
@login_required
def export_excel():
    user_id, role, is_admin = get_current_user()
    conn = database.get_connection()
    param = "%s" if database.USE_MYSQL else "?"
    if is_admin:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
    else:
        df = pd.read_sql_query(f"SELECT * FROM predictions WHERE user_id = {user_id}", conn)
    conn.close()
    
    file_name = "Telemetry_Predictions.xlsx"
    df.to_excel(file_name, index=False)
    return send_file(file_name, as_attachment=True)

# ---------------- RUN APPLICATION ----------------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    host_val = os.environ.get("HOST", "127.0.0.1")
    port_val = int(os.environ.get("PORT", 5000))
    app.run(
        debug=debug_mode,
        host=host_val,
        port=port_val
    )

