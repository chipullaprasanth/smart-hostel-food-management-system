-- MySQL Schema setup for Smart Hostel Food Management

CREATE DATABASE IF NOT EXISTS hostel_db;
USE hostel_db;

-- Predictions Log
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    day VARCHAR(15) NOT NULL,
    month VARCHAR(15) NOT NULL,
    temperature FLOAT,
    rainfall FLOAT,
    humidity INT,
    wind_speed FLOAT,
    holiday VARCHAR(5),
    festival VARCHAR(5),
    exam VARCHAR(5),
    sports_event VARCHAR(5),
    breakfast_menu VARCHAR(50),
    lunch_menu VARCHAR(50),
    dinner_menu VARCHAR(50),
    previous_attendance INT,
    previous_waste FLOAT,
    food_rating FLOAT,
    predicted_attendance INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Credentials
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);

-- Feedback Logs
CREATE TABLE IF NOT EXISTS feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50),
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily Attendance Log
CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date VARCHAR(20) UNIQUE NOT NULL,
    students INT NOT NULL
);

-- Store Inventory
CREATE TABLE IF NOT EXISTS inventory (
    id INT AUTO_INCREMENT PRIMARY KEY,
    item VARCHAR(50) UNIQUE NOT NULL,
    quantity VARCHAR(30) NOT NULL
);

-- Donation logs
CREATE TABLE IF NOT EXISTS donations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization VARCHAR(100),
    food VARCHAR(100),
    quantity VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Financial Expenses
CREATE TABLE IF NOT EXISTS expenses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date VARCHAR(20),
    amount DOUBLE,
    description VARCHAR(255)
);

-- Daily waste calculations
CREATE TABLE IF NOT EXISTS waste_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date VARCHAR(20) UNIQUE,
    prepared FLOAT,
    consumed FLOAT,
    waste FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Default admin user registration (admin / admin123 hashed representation)
-- Hashed password representation is handled dynamically during runtime setup.
