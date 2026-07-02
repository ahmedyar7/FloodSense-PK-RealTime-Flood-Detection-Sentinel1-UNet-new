import os
import json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class FloodAI:
    def __init__(self):
        self.gemini_enabled = False
        
        # Gemini Init (New Primary)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and len(gemini_key) > 10 and gemini_key != "your-gemini-key":
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=gemini_key.strip())
                self.gemini_model_name = "gemini-1.5-flash"
                self.gemini_enabled = True
            except Exception as e:
                print(f"Gemini configuration failed: {e}")
                self.gemini_enabled = False

        # Groq Init (Secondary)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and "gsk_" in groq_key:
            try:
                self.groq_client = Groq(api_key=groq_key.strip())
                self.groq_enabled = True
            except Exception as e:
                print(f"Groq configuration failed: {e}")
                self.groq_enabled = False
        else:
            self.groq_enabled = False

        self.enabled = self.gemini_enabled or self.groq_enabled
        if not self.enabled:
            print("No valid AI API keys (Gemini/Groq) found. AI insights will be simulated.")

    def calculate_defensible_risk(self, d, river_data):
        """
        Derives a risk score (1-10) using weighted multi-factor analysis.
        """
        # 1. Flood Extent Weight (40%)
        f_score = min(10, (d.get('flood_pct_current', 0) / 5) * 10) 
        
        # 2. Delta vs 2010 Weight (30%)
        # If today is worse than 2010, this spikes the score.
        delta = d.get('flood_pct_current', 0) - d.get('flood_pct_2010', 0)
        d_score = 5 + (delta * 2) # Base 5, +2 for every % above 2010
        d_score = max(0, min(10, d_score))
        
        # 3. Hydraulic Weight (30%)
        h_score = 1
        status = d.get('river_status', 'UNKNOWN')
        if status == 'EXTREME': h_score = 10
        elif status == 'HIGH': h_score = 7
        elif status == 'NORMAL': h_score = 2
        
        # Weighted sum
        raw_risk = (f_score * 0.4) + (d_score * 0.3) + (h_score * 0.3)
        return round(max(1, min(10, raw_risk)), 1)

    def generate_insights(self, district_data, river_data):
        """
        Generates grounded strategic insights with numerical evidence.
        """
        d = district_data[0] # Focus on current selection
        risk_score = self.calculate_defensible_risk(d, river_data)
        
        prompt = f"""
        Act as a Flood Intelligence Officer. Analyze this Pakistan flood data and provide a structured operational report.
        
        MANDATORY RULES:
        - REFERENCE NUMBERS (flood %, inflow, delta) in every point.
        - No generic advice like "monitor closely".
        - If data is "UNKNOWN", state it as a "Data Gap" that reduces confidence.
        
        DATA CONTEXT:
        - District: {d['district']}
        - Current Flood %: {d['flood_pct_current']:.2f}%
        - 2010 Historical %: {d['flood_pct_2010']:.2f}%
        - Delta vs 2010: {(d['flood_pct_current'] - d['flood_pct_2010']):.2f}%
        - River Status: {d['river_status']}
        - Inflow at nearest station: {json.dumps(river_data[:3], indent=2)}
        - Calculated Risk Score: {risk_score}/10
        
        REPORT STRUCTURE:
        1. [SITUATION SUMMARY] - 1 sentence summary using flood %.
        2. [HYDRAULIC ANALYSIS] - Link river inflow to inundation.
        3. [HISTORICAL BENCHMARK] - Compare today to 2010 with delta.
        4. [OPERATIONAL ACTIONS] - 2 specific actions for this district.
        5. [CONFIDENCE] - High/Med/Low based on UNKNOWN statuses.
        """
        
        if self.gemini_enabled:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                print(f"Gemini error: {e}")

        return self._get_fallback_insights(district_data, river_data)

    def _try_gemini(self, prompt: str):
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key or gemini_key == "your-gemini-key" or len(gemini_key) <= 10:
            return None
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key.strip())
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Gemini error: {e}")
            return None

    def _get_fallback_insights(self, district_data, river_data):
        # Fallback logic if API key is missing
        # We handle potentially missing flood_pct keys
        critical_districts = [d['district'] for d in district_data if (d.get('flood_pct_current') or d.get('flood_pct') or 0) > 5]
        high_flow_stations = [s['station'] for s in river_data if s.get('status') == 'HIGH']
        
        insights = "### AI Strategic Insights (Simulated)\n\n"
        if critical_districts:
            insights += f"1. **Deployment Priority:** Immediate relief teams should be prioritized for {', '.join(critical_districts[:3])} due to high inundation levels.\n"
        else:
            insights += "1. **Monitoring:** No districts currently exceed the critical 5% inundation threshold, but continuous monitoring is advised.\n"
            
        if high_flow_stations:
             insights += f"2. **Logistics Risk:** High river flows detected at {', '.join(high_flow_stations[:2])} may threaten nearby transport links.\n"
        else:
             insights += "2. **Infrastructure:** River flows are currently within normal ranges; however, check for localized urban flooding.\n"
             
        insights += "3. **Predictive Alert:** Maintain readiness in downstream Sindh districts as upper Indus basin shows moderate snowmelt/rainfall runoff."
        
        return insights

    def save_alert_to_json(self, alert_data):
        os.makedirs("data/json", exist_ok=True)
        path = "data/json/alerts.json"
        
        alerts = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                try:
                    alerts = json.load(f)
                except:
                    alerts = []
                    
        alert_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alerts.append(alert_data)
        
        with open(path, 'w') as f:
            json.dump(alerts, f, indent=4)
