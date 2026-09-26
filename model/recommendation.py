"""
Food Recommendation Engine for Smart Hostel Food Waste Management System.
Computes optimal food quantities (Rice, Dal, Curry, Chapati) and seasonal menu recommendations
based on Attendance (500-1000 students), Season, Temperature, Weather, Month, and Festival status.
"""

def derive_season(month_name, temp):
    month = str(month_name).capitalize()
    if month in ["December", "January", "February"]:
        return "Winter"
    elif month in ["March", "April", "May"]:
        return "Summer"
    elif month in ["June", "July", "August", "September"]:
        return "Monsoon"
    else:
        return "Autumn"

def derive_weather_condition(temp, rainfall, humidity):
    if rainfall > 5.0:
        return "Rainy / Stormy"
    elif temp > 35:
        return "Hot & Sunny"
    elif temp < 20:
        return "Cold & Chilly"
    elif humidity > 80:
        return "Humid"
    else:
        return "Pleasant / Clear"

def generate_recommendations(attendance, season, temp, rainfall, humidity, festival="Normal Day", b_menu=None, l_menu=None, d_menu=None):
    """
    Computes recommended food quantities (in kg / count) for 500-1000 student capacity
    and returns a seasonal/weather-tailored menu recommendation.
    """
    students = max(100, int(attendance))
    
    # Base per-student consumption factors
    base_rice_per_student = 0.25      # kg
    base_dal_per_student = 0.12       # kg
    base_curry_per_student = 0.15     # kg
    base_chapati_per_student = 2.5    # pieces
    
    season_lower = str(season).lower()
    weather_cond = derive_weather_condition(temp, rainfall, humidity)
    
    rice_mult = 1.0
    dal_mult = 1.0
    curry_mult = 1.0
    chapati_mult = 1.0
    
    if "winter" in season_lower or temp < 20:
        rice_mult *= 0.95
        dal_mult *= 1.08
        curry_mult *= 1.10
        chapati_mult *= 1.10
    elif "summer" in season_lower or temp > 34:
        rice_mult *= 1.08
        dal_mult *= 1.02
        curry_mult *= 0.92
        chapati_mult *= 0.92
    elif "monsoon" in season_lower or rainfall > 5.0:
        rice_mult *= 0.98
        dal_mult *= 1.05
        curry_mult *= 1.12
        chapati_mult *= 1.05
        
    # Festival Multipliers
    fest_lower = str(festival).lower()
    if "festival" in fest_lower or "special" in fest_lower:
        rice_mult *= 1.12
        curry_mult *= 1.15
        chapati_mult *= 1.08
    elif "holiday" in fest_lower or "exam" in fest_lower:
        rice_mult *= 0.90
        dal_mult *= 0.92
        curry_mult *= 0.90
        chapati_mult *= 0.90

    # Calculate recommended totals
    rec_rice = round(students * base_rice_per_student * rice_mult, 1)
    rec_dal = round(students * base_dal_per_student * dal_mult, 1)
    rec_curry = round(students * base_curry_per_student * curry_mult, 1)
    rec_chapati = int(round(students * base_chapati_per_student * chapati_mult))
    
    # Seasonally Suitable Menu Recommendations & Rationale
    if "festival" in fest_lower:
        suggested_breakfast = "Puri Bhaji & Kheer"
        suggested_lunch = "Special Veg Biryani & Paneer Butter Masala"
        suggested_dinner = "Jeera Rice, Malai Kofta & Gulab Jamun"
        rationale = "Festive Special Menu: Increased rice and paneer portions to match high festive mess attendance and celebration demand."
    elif "winter" in season_lower or temp < 20:
        suggested_breakfast = "Hot Aloo Paratha & Buttered Tea"
        suggested_lunch = "Rajma Chawal & Mix Veg Fry"
        suggested_dinner = "Hot Roti, Sev Tamatar & Dal Fry"
        rationale = "Winter Season Menu: Calorie-dense, warm chapati and protein-rich dal satisfy higher thermal energy demands and boost immunity."
    elif "summer" in season_lower or temp > 34:
        suggested_breakfast = "Idli Sambar & Coconut Chutney"
        suggested_lunch = "Curd Rice, Lemon Rice & Cucumber Salad"
        suggested_dinner = "Light Khichdi Kadhi & Soft Roti"
        rationale = "Summer Season Menu: Light, hydrating rice and fermented curd items prevent food spoilage and digestive heaviness in high heat."
    elif "monsoon" in season_lower or rainfall > 5.0:
        suggested_breakfast = "Hot Poha Upma & Crispy Pakoras"
        suggested_lunch = "Chole Bhature & Jeera Rice"
        suggested_dinner = "Warm Roti, Mix Veg Green Curry & Dal Tadka"
        rationale = "Monsoon Season Menu: Warm, freshly prepared spicy gravies and hot flatbreads minimize bacterial risks and food moisture spoilage."
    else:
        suggested_breakfast = b_menu or "Masala Dosa & Sambar"
        suggested_lunch = l_menu or "Dal Chawal Sabzi & Chapati"
        suggested_dinner = d_menu or "Roti Green Curry & Dal Fry"
        rationale = "Balanced Standard Menu: Optimal nutritional balance for moderate weather operational conditions."

    return {
        "attendance": students,
        "rice_kg": rec_rice,
        "dal_kg": rec_dal,
        "curry_kg": rec_curry,
        "chapati_count": rec_chapati,
        "season": season,
        "weather_condition": weather_cond,
        "festival": festival,
        "suggested_menu": {
            "breakfast": suggested_breakfast,
            "lunch": suggested_lunch,
            "dinner": suggested_dinner
        },
        "menu_rationale": rationale
    }
