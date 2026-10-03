# PredictScore - live tennis scores + win predictions (ATP + WTA)

## Files (upload ALL to your GitHub repo, same folder level)
app.py, scoreboard.py, live_tennis.py, tennis_model.py, requirements.txt, README.md
Optional: .streamlit/config.toml  (dark theme; if you skip it, choose Dark in the app menu > Settings > Theme)

## Turn on REAL live matches (free)
1. Get a free key (no card): https://livetennisapi.com/subscribe/free
2. Streamlit Cloud > your app > Settings > Secrets, paste:
       LIVETENNIS_API_KEY = "your-key-here"
3. Save. The app reloads and the Live Scores tab shows real matches.
Free plan = 100 requests/day. The app caches for 2 minutes, so press Refresh only when needed.
Without a key the tab shows demo matches.

## Data
Ratings: Jeff Sackmann / Tennis Abstract (CC BY-NC-SA 4.0, non-commercial).
Live scores: Live Tennis API (free tier). Independent project, not affiliated with Sofascore.
