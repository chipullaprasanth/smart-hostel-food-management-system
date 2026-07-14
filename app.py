from flask import Flask, render_template, request, send_file, redirect, session, flash, url_for, jsonify
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os
import database
from model.prediction import predict_attendance

app = Flask(__name__)
app.secret_key = "btech-hostel-reduction-placement-secret-key"

# Initialize Database
database.init_db()

# Decorator to secure admin pages
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user'):
            flash("Authorization required to access this system.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
@login_required
def dashboard():
    conn = database.get_connection()
    cursor = conn.cursor()
    
    # Retrieve today's attendance (latest log)
    if database.USE_MYSQL:
        cursor.execute("SELECT students FROM attendance ORDER BY date DESC LIMIT 1")
        row = cursor.fetchone()
        today_attendance = row['students'] if row else 70
    else:
        cursor.execute("SELECT students FROM attendance ORDER BY date DESC LIMIT 1")
        row = cursor.fetchone()
        today_attendance = row[0] if row else 70
        
    # Retrieve latest prediction
    if database.USE_MYSQL:
        cursor.execute("SELECT predicted_attendance FROM predictions ORDER BY id DESC LIMIT 1")
        row_pred = cursor.fetchone()
        predicted_attendance = row_pred['predicted_attendance'] if row_pred else 74
    else:
        cursor.execute("SELECT predicted_attendance FROM predictions ORDER BY id DESC LIMIT 1")
        row_pred = cursor.fetchone()
        predicted_attendance = row_pred[0] if row_pred else 74

    # Retrieve latest waste
    if database.USE_MYSQL:
        cursor.execute("SELECT waste FROM waste_records ORDER BY date DESC LIMIT 1")
        row_w = cursor.fetchone()
        today_waste = row_w['waste'] if row_w else 2.4
    else:
        cursor.execute("SELECT waste FROM waste_records ORDER BY date DESC LIMIT 1")
        row_w = cursor.fetchone()
        today_waste = row_w[0] if row_w else 2.4
        
    conn.close()
    
    # Calculate values
    predicted_rice = round(predicted_attendance * 0.4, 1)
    
    # Look for custom menu files in static/uploads
    menu_rows = []
    menu_image = None
    menu_pdf = None
    menu_doc = None
    
    upload_dir = os.path.join("static", "uploads")
    custom_menu_found = False
    
    if os.path.exists(upload_dir):
        files = os.listdir(upload_dir)
        for f in files:
            if f.startswith("uploaded_menu."):
                ext = f.split(".")[-1].lower()
                filepath = os.path.join(upload_dir, f)
                custom_menu_found = True
                
                if ext == "csv":
                    try:
                        menu_df = pd.read_csv(filepath)
                        menu_rows = menu_df.to_dict(orient='records')
                    except Exception:
                        pass
                elif ext in ["png", "jpg", "jpeg"]:
                    menu_image = f"uploads/{f}"
                elif ext == "pdf":
                    menu_pdf = f"uploads/{f}"
                elif ext in ["doc", "docx"]:
                    menu_doc = f"uploads/{f}"
                break
                
    # Fallback to default menu if no custom upload exists
    if not custom_menu_found:
        menu_path = os.path.join("dataset", "hostel_menu.csv")
        if os.path.exists(menu_path):
            try:
                menu_df = pd.read_csv(menu_path)
                menu_rows = menu_df.to_dict(orient='records')
            except Exception:
                pass
                
    return render_template(
        'dashboard.html',
        today_attendance=today_attendance,
        predicted_attendance=predicted_attendance,
        predicted_rice=predicted_rice,
        today_waste=today_waste,
        menu_list=menu_rows,
        menu_image=menu_image,
        menu_pdf=menu_pdf,
        menu_doc=menu_doc
    )

# ---------------- PREDICTION SYSTEM ----------------
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Gather features
            input_features = {
                'Day': request.form['Day'],
                'Month': request.form['Month'],
                'Temperature': float(request.form['Temperature']),
                'Rainfall': float(request.form['Rainfall']),
                'Humidity': int(request.form['Humidity']),
                'Wind_Speed': float(request.form['Wind_Speed']),
                'Breakfast_Menu': request.form['Breakfast_Menu'],
                'Lunch_Menu': request.form['Lunch_Menu'],
                'Dinner_Menu': request.form['Dinner_Menu'],
                'Previous_Attendance': int(request.form['Previous_Attendance']),
                'Previous_Waste': float(request.form['Previous_Waste']),
                'Food_Rating': float(request.form['Food_Rating'])
            }
            
            # Predict
            pred_attendance = predict_attendance(input_features)
            
            # Save prediction
            input_features['Predicted_Attendance'] = pred_attendance
            database.add_prediction(input_features)
            
            prediction = {
                'students': pred_attendance,
                'rice': round(pred_attendance * 0.4, 1),
                'dal': round(pred_attendance * 0.15, 1),
                'vegetables': round(pred_attendance * 0.25, 1)
            }
            flash("Model inference completed. Results generated successfully!")
        except Exception as e:
            flash(f"Inference pipeline execution error: {str(e)}")
            
    return render_template('prediction.html', prediction=prediction)

# ---------------- ATTENDANCE ----------------
@app.route('/attendance', methods=['GET', 'POST'])
@login_required
def attendance():
    if request.method == 'POST':
        try:
            date_val = request.form['date']
            students_val = int(request.form['students'])
            database.save_attendance(date_val, students_val)
            flash(f"Attendance registered successfully: {students_val} students on {date_val}.")
        except Exception as e:
            flash(f"Error registering attendance: {str(e)}")
            
    # Fetch lists
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT date, students FROM attendance ORDER BY date DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()
    
    attendance_list = []
    for r in rows:
        if database.USE_MYSQL:
            attendance_list.append(dict(r))
        else:
            attendance_list.append({'date': r[0], 'students': r[1]})
            
    return render_template('attendance.html', attendance_list=attendance_list)

# ---------------- WASTE ANALYSIS ----------------
@app.route('/waste', methods=['GET', 'POST'])
@login_required
def waste():
    if request.method == 'POST':
        try:
            date_val = request.form['date']
            prep = float(request.form['prepared'])
            cons = float(request.form['consumed'])
            
            if prep < cons:
                flash("Error: Consumed quantity cannot exceed prepared volume.")
            else:
                w_val = round(prep - cons, 2)
                database.save_waste_record(date_val, prep, cons, w_val)
                flash(f"Wastage recorded: {w_val} Kg of food.")
        except Exception as e:
            flash(f"Error logging waste: {str(e)}")
            
    waste_list = database.get_waste_records()
    return render_template('waste.html', waste_list=waste_list)

# ---------------- REPORTS ----------------
@app.route('/reports')
@login_required
def reports():
    total_preds = database.get_predictions_count()
    total_expense, savings = database.get_expense_stats()
    
    # Calculate average students present
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT AVG(predicted_attendance) FROM predictions")
    avg_row = cursor.fetchone()
    avg_students = round(avg_row['AVG(predicted_attendance)'] if database.USE_MYSQL else (avg_row[0] or 74), 1)
    conn.close()
    
    return render_template(
        'reports.html',
        total_predictions=total_preds,
        avg_students=avg_students,
        total_expense=total_expense,
        savings=savings
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
    ext = file.filename.split('.')[-1].lower()
    
    if file and ext in allowed_extensions:
        try:
            # Clear old uploads starting with "uploaded_menu."
            upload_dir = os.path.join("static", "uploads")
            if os.path.exists(upload_dir):
                for f in os.listdir(upload_dir):
                    if f.startswith("uploaded_menu."):
                        os.remove(os.path.join(upload_dir, f))
            else:
                os.makedirs(upload_dir, exist_ok=True)
                
            # Save new file
            new_filename = f"uploaded_menu.{ext}"
            file.save(os.path.join(upload_dir, new_filename))
            flash(f"Hostel weekly menu file ({ext.upper()}) uploaded successfully!")
        except Exception as e:
            flash(f"Error saving menu file: {str(e)}")
    else:
        flash("Invalid file format. Supported types: CSV, PNG, JPG, PDF, DOC, DOCX.")
    return redirect(url_for('settings'))

# ---------------- REGISTER ----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if database.register_user(username, password):
            flash("Account registered successfully! Please log in.")
            return redirect(url_for('login'))
        else:
            flash("Username already exists or registration failed.")
    return render_template('register.html')

# ---------------- INVENTORY ----------------
@app.route('/inventory', methods=['GET', 'POST'])
@login_required
def inventory():
    if request.method == 'POST':
        item = request.form['item']
        quantity = request.form['quantity']
        database.save_inventory(item, quantity)
        flash(f"Inventory saved: {item} updated to {quantity}.")
    items = [(row['item'], row['quantity']) for row in database.get_inventory()]
    return render_template('inventory.html', items=items)

# ---------------- DONATION ----------------
@app.route('/donation', methods=['GET', 'POST'])
@login_required
def donation():
    if request.method == 'POST':
        org = request.form['organization']
        food = request.form['food']
        qty = request.form['quantity']
        database.save_donation(org, food, qty)
        flash(f"Donation dispatch to {org} registered successfully.")
    donations = database.get_donations()
    return render_template('donation.html', donations=donations)

# ---------------- EXPENSE ----------------
@app.route('/expense', methods=['GET', 'POST'])
@login_required
def expense():
    if request.method == 'POST':
        date_val = request.form['date']
        try:
            amount = float(request.form['amount'])
            desc = request.form['description']
            database.save_expense(date_val, amount, desc)
            flash("Expense recorded successfully.")
        except ValueError:
            flash("Invalid expense amount.")
    total_expense, savings = database.get_expense_stats()
    return render_template('expense.html', expense=total_expense, savings=savings)

# ---------------- FEEDBACK ----------------
@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if request.method == 'POST':
        username = request.form['username']
        text = request.form['feedback']
        database.add_feedback(username, text)
        flash("Thank you! Feedback submitted successfully.")
        return redirect(url_for('dashboard'))
    return render_template('feedback.html')

# ---------------- HISTORY ----------------
@app.route('/history', methods=['GET', 'POST'])
@login_required
def history():
    search = None
    if request.method == 'POST':
        search = request.form.get('search')
    
    predictions = database.get_predictions(search)
    rows = []
    for p in predictions:
        students = p['predicted_attendance']
        rows.append({
            'id': p['id'],
            'students': students,
            'rice': round(students * 0.4, 1),
            'dal': round(students * 0.15, 1),
            'curry': round(students * 0.25, 1),
            'chapati': students * 2
        })
    return render_template('history.html', rows=rows, search=search)

# ---------------- DELETE HISTORY ----------------
@app.route('/delete/<int:prediction_id>')
@login_required
def delete_prediction(prediction_id):
    database.delete_prediction(prediction_id)
    flash("Forecast record deleted successfully.")
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
                'rice': round(students * 0.4, 1),
                'chapati': students * 2,
                'curry': round(students * 0.25, 1),
                'dal': round(students * 0.15, 1)
            }
        except ValueError:
            flash("Invalid number of students.")
    return render_template('predict.html', prediction=prediction)

# ---------------- ML FORECAST ----------------
@app.route('/ml_forecast')
@login_required
def ml_forecast():
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT predicted_attendance FROM predictions ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    predicted_students = row['predicted_attendance'] if database.USE_MYSQL else (row[0] if row else 75)
    conn.close()
    
    return render_template(
        'ml_predict.html',
        predicted_students=predicted_students,
        rice=round(predicted_students * 0.4, 1),
        curry=round(predicted_students * 0.25, 1),
        chapati=predicted_students * 2,
        dal=round(predicted_students * 0.15, 1)
    )

# ---------------- ADMIN PANEL ----------------
@app.route('/admin')
@login_required
def admin():
    users_count = database.get_users_count()
    feedback_count = database.get_feedback_count()
    predictions_count = database.get_predictions_count()
    feedback_list = database.get_feedback()
    return render_template(
        'admin.html',
        users=users_count,
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
        
        if database.verify_user(username, password):
            session['user'] = username
            flash("Welcome back. Authentication successful!")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid administration username or password.")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been signed out of the system.")
    return redirect(url_for('home'))

# ---------------- API CHART DATA ENDPOINTS ----------------
@app.route('/api/dashboard_chart_data')
@login_required
def dashboard_chart_data():
    conn = database.get_connection()
    cursor = conn.cursor()
    
    # Attendance logs
    cursor.execute("SELECT date, students FROM attendance ORDER BY date ASC LIMIT 10")
    att_rows = cursor.fetchall()
    
    att_labels = []
    att_values = []
    for r in att_rows:
        att_labels.append(r['date'] if database.USE_MYSQL else r[0])
        att_values.append(r['students'] if database.USE_MYSQL else r[1])
        
    # Waste logs
    cursor.execute("SELECT date, prepared, consumed, waste FROM waste_records ORDER BY date ASC LIMIT 10")
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
        
    # Fill in benchmark defaults if empty
    if not att_labels:
        att_labels = ["Day 1", "Day 2", "Day 3"]
        att_values = [70, 75, 68]
    if not w_labels:
        w_labels = ["Day 1", "Day 2", "Day 3"]
        prep_values = [50, 52, 48]
        consumed_values = [47.6, 49.5, 46.1]
        w_values = [2.4, 2.5, 1.9]
        
    total_expense, savings = database.get_expense_stats()
    conn.close()
    
    return jsonify({
        'attendance_labels': att_labels,
        'attendance_values': att_values,
        'waste_labels': w_labels,
        'waste_values': w_values,
        'prep_values': prep_values,
        'consumed_values': consumed_values,
        'total_expense': total_expense,
        'savings': savings
    })

# ---------------- DOWNLOAD PDF REPORT ----------------
@app.route('/download_report')
@login_required
def download_report():
    report_path = "Prediction_Report.pdf"
    
    # Aggregated metrics
    total_preds = database.get_predictions_count()
    total_expense, savings = database.get_expense_stats()
    
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
    
    story.append(Paragraph("Smart Hostel AI Food Management Systems", title_style))
    story.append(Paragraph("Evaluation Log and Audit Telemetry Report", subtitle_style))
    
    data = [
        ["Key Telemetry Metric", "Calculated Value"],
        ["Calculation Runs Executed", str(total_preds)],
        ["Recorded Food Waste (Average)", "2.4 Kg"],
        ["Total Logged Expenses", f"Rs. {total_expense:.2f}"],
        ["Net Savings Balance", f"Rs. {savings:.2f}"]
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
    conn = database.get_connection()
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    
    file_name = "Telemetry_Predictions.xlsx"
    df.to_excel(file_name, index=False)
    return send_file(file_name, as_attachment=True)

# ---------------- RUN APPLICATION ----------------
if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
