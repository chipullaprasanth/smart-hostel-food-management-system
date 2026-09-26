import os
import sys
import unittest
import tempfile
from io import BytesIO

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app and database
import app
import database

class ApplicationHardeningTestSuite(unittest.TestCase):
    def setUp(self):
        app.app.config['TESTING'] = True
        app.app.config['WTF_CSRF_ENABLED'] = True
        self.client = app.app.test_client()
        database.init_db()

    def test_01_python_syntax_imports(self):
        """1. Python syntax & import checks"""
        self.assertTrue(hasattr(app, 'app'))
        self.assertTrue(hasattr(database, 'init_db'))
        self.assertTrue(hasattr(database, 'register_user'))

    def test_02_app_startup(self):
        """2. Application startup test"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_03_db_initialization(self):
        """3. Database initialization test"""
        user_cnt = database.get_users_count()
        self.assertGreaterEqual(user_cnt, 1)

    def test_04_authentication(self):
        """4. Authentication test"""
        # Obtain CSRF token from login page
        res = self.client.get('/login')
        csrf_token = self.get_csrf_token(res.data.decode('utf-8'))
        
        # Test valid login with seeded admin
        res_post = self.client.post('/login', data={
            'username': 'admin',
            'password': 'admin123',
            'csrf_token': csrf_token
        }, follow_redirects=True)
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Dashboard', res_post.data)

    def test_05_normal_user_authorization(self):
        """5. Normal user authorization test"""
        # Register normal user
        res_reg_page = self.client.get('/register')
        csrf_token = self.get_csrf_token(res_reg_page.data.decode('utf-8'))
        
        username = "normal_user_test"
        # Attempt to pass role='admin' in POST data - should be ignored!
        self.client.post('/register', data={
            'username': username,
            'password': 'user123',
            'role': 'admin',
            'csrf_token': csrf_token
        }, follow_redirects=True)

        # Login as normal user
        res_login_page = self.client.get('/login')
        csrf_token = self.get_csrf_token(res_login_page.data.decode('utf-8'))
        self.client.post('/login', data={
            'username': username,
            'password': 'user123',
            'csrf_token': csrf_token
        }, follow_redirects=True)

        # Verify normal user cannot access /admin
        res_admin = self.client.get('/admin', follow_redirects=True)
        self.assertIn(b'Admin privileges required', res_admin.data)

        # Verify role in DB is strictly 'user'
        users = database.get_all_users()
        norm_u = next(u for u in users if u['username'] == username)
        self.assertEqual(norm_u['role'], 'user')

    def test_06_admin_authorization(self):
        """6. Admin authorization test"""
        res_login_page = self.client.get('/login')
        csrf_token = self.get_csrf_token(res_login_page.data.decode('utf-8'))
        self.client.post('/login', data={
            'username': 'admin',
            'password': 'admin123',
            'csrf_token': csrf_token
        }, follow_redirects=True)

        res_admin = self.client.get('/admin')
        self.assertEqual(res_admin.status_code, 200)

    def test_07_prediction_test(self):
        """7. Prediction test"""
        self.login_user('admin', 'admin123')
        res_page = self.client.get('/predict')
        csrf_token = self.get_csrf_token(res_page.data.decode('utf-8'))

        res_post = self.client.post('/predict', data={
            'date': '2026-08-11',
            'Day': 'Tuesday',
            'Month': 'August',
            'Season': 'Monsoon',
            'Temperature': '28.5',
            'Rainfall': '12.0',
            'Humidity': '75',
            'Wind_Speed': '14.0',
            'Breakfast_Menu': 'Idli',
            'Lunch_Menu': 'Rice & Dal',
            'Dinner_Menu': 'Roti & Sabzi',
            'Previous_Attendance': '750',
            'Previous_Waste': '14.2',
            'Food_Rating': '4.2',
            'Festival': 'Normal Day',
            'csrf_token': csrf_token
        }, follow_redirects=True)
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'ML Forecasting &amp; Food Recommendation completed', res_post.data)

    def test_08_attendance_test(self):
        """8. Attendance test"""
        self.login_user('admin', 'admin123')
        res_page = self.client.get('/attendance')
        csrf_token = self.get_csrf_token(res_page.data.decode('utf-8'))

        res_post = self.client.post('/attendance', data={
            'date': '2026-08-11',
            'students': '790',
            'csrf_token': csrf_token
        }, follow_redirects=True)
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Attendance registered successfully', res_post.data)

    def test_09_waste_test(self):
        """9. Waste test"""
        self.login_user('admin', 'admin123')
        res_page = self.client.get('/waste')
        csrf_token = self.get_csrf_token(res_page.data.decode('utf-8'))

        res_post = self.client.post('/waste', data={
            'date': '2026-08-11',
            'prepared': '420.0',
            'consumed': '405.0',
            'csrf_token': csrf_token
        }, follow_redirects=True)
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Wastage recorded', res_post.data)

    def test_10_inventory_test(self):
        """10. Inventory test"""
        self.login_user('admin', 'admin123')
        res_page = self.client.get('/inventory')
        csrf_token = self.get_csrf_token(res_page.data.decode('utf-8'))

        res_post = self.client.post('/inventory', data={
            'item': 'Basmati Rice',
            'quantity': '2000 Kg',
            'csrf_token': csrf_token
        }, follow_redirects=True)
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Inventory saved', res_post.data)

    def test_11_expense_test(self):
        """11. Expense test"""
        self.login_user('admin', 'admin123')
        res_page = self.client.get('/expense')
        csrf_token = self.get_csrf_token(res_page.data.decode('utf-8'))

        res_post = self.client.post('/expense', data={
            'date': '2026-08-11',
            'amount': '1500.00',
            'description': 'Fresh vegetables procurement',
            'csrf_token': csrf_token
        }, follow_redirects=True)
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Expense recorded successfully', res_post.data)

    def test_12_donation_removed(self):
        """12. Donation feature removal verification"""
        self.login_user('admin', 'admin123')
        res_page = self.client.get('/donation')
        self.assertEqual(res_page.status_code, 404)

    def test_13_feedback_test(self):
        """13. Feedback test"""
        self.login_user('admin', 'admin123')
        res_page = self.client.get('/feedback')
        csrf_token = self.get_csrf_token(res_page.data.decode('utf-8'))

        res_post = self.client.post('/feedback', data={
            'username': 'admin',
            'feedback': 'Great food quality and low waste today!',
            'csrf_token': csrf_token
        }, follow_redirects=True)
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Thank you! Feedback submitted successfully', res_post.data)

    def test_14_history_delete_authorization(self):
        """14. History/delete authorization test"""
        self.login_user('admin', 'admin123')
        preds = database.get_predictions(is_admin=True)
        if preds:
            pred_id = preds[0]['id']
            res_del = self.client.get(f'/delete/{pred_id}', follow_redirects=True)
            self.assertIn(b'Forecast record deleted successfully', res_del.data)

    def test_15_file_upload_test(self):
        """15. File upload test"""
        self.login_user('admin', 'admin123')
        res_page = self.client.get('/settings')
        csrf_token = self.get_csrf_token(res_page.data.decode('utf-8'))

        data = {
            'menu_file': (BytesIO(b"Day,Breakfast,Lunch,Dinner\nMonday,Idli,Rice,Roti"), 'uploaded_menu.csv'),
            'csrf_token': csrf_token
        }
        res_post = self.client.post('/upload_menu', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Weekly hostel menu uploaded successfully', res_post.data)

    def test_16_report_generation_test(self):
        """16. Report generation test"""
        self.login_user('admin', 'admin123')
        res = self.client.get('/download_report')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'application/pdf')

    def test_17_excel_export_test(self):
        """17. Excel export test"""
        self.login_user('admin', 'admin123')
        res = self.client.get('/export_excel')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    def test_18_api_endpoint_test(self):
        """18. API endpoint test"""
        self.login_user('admin', 'admin123')
        res = self.client.get('/api/dashboard_chart_data')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertIn('attendance_labels', json_data)
        self.assertIn('waste_labels', json_data)
        self.assertIn('food_waste_types', json_data)
        self.assertIn('food_cost_sections', json_data)
        self.assertIn('global_benchmarks', json_data)

    def get_csrf_token(self, html_text):
        if 'name="csrf_token" value="' in html_text:
            return html_text.split('name="csrf_token" value="')[1].split('"')[0]
        return ''

    def login_user(self, username, password):
        res_login_page = self.client.get('/login')
        csrf_token = self.get_csrf_token(res_login_page.data.decode('utf-8'))
        self.client.post('/login', data={
            'username': username,
            'password': password,
            'csrf_token': csrf_token
        }, follow_redirects=True)

if __name__ == '__main__':
    unittest.main()
