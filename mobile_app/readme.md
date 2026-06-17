# FloodSense-PK Mobile App

A simple mobile application for flood alerts, built with **Flet** (Flutter for Python).

## Features
- **User Authentication:** Secure login and registration using SQLite.
- **District Selection:** Choose your district to receive personalized alerts.
- **Flood Monitoring:** Real-time river data and AI-generated risk reports.
- **Modern UI:** Clean, dark-themed interface.

## Setup Instructions

1. **Install Dependencies:**
   Ensure you have the required libraries installed:
   ```bash
   pip install flet requests python-dotenv groq google-generativeai
   ```

2. **Run the App:**
   Navigate to the `mobile_app` folder and run the `main.py` script:
   ```bash
   cd mobile_app
   python main.py
   ```

3. **Packaging for Mobile:**
   Flet apps can be packaged for Android or iOS. For more information, visit the [Flet documentation](https://flet.dev/docs/guides/python/packaging-app-for-distribution).

## File Structure
- `main.py`: The entry point for the Flet application.
- `database.py`: Handles SQLite user authentication.
- `alert_engine.py`: Fetches real-time data and generates AI alerts.
- `districts.json`: List of available districts.
- `users.db`: SQLite database (auto-generated).
