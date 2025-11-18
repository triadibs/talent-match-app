# Talent Match Intelligence System

AI-powered talent matching and succession planning platform.

## Features
- 🎯 Dynamic job vacancy creation
- 🤖 AI-generated job profiles (via OpenRouter)
- 📊 Multi-dimensional talent matching (7 TGVs, 50+ TVs)
- 📈 Interactive analytics dashboard
- 🔍 Detailed TV-level match insights

## Setup

### Prerequisites
- Python 3.9+
- PostgreSQL (Supabase)
- OpenRouter API key (optional, fallback available)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/talent-match-app
cd talent-match-app

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your credentials
```

### Database Setup
```sql
-- Run your SQL scripts from Step 2
-- Create tables: talent_benchmarks, employees, etc.
```

### Run Locally
```bash
streamlit run app.py
```

## Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Add secrets in Streamlit Cloud settings
5. Deploy!

## Project Structure
```
talent-match-app/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── config.py             # Configuration
├── utils/
│   ├── database.py       # Database operations
│   ├── llm.py           # LLM integration
│   └── visualizations.py # Charts & plots
└── README.md
```

## Usage

### Create Vacancy
1. Go to "Create Vacancy" page
2. Fill in role details
3. Select 2-5 high performers as benchmarks
4. Optionally adjust TGV weights
5. Click "Create Vacancy & Run Matching"

### View Results
1. See AI-generated job profile
2. Browse top candidates ranked by match rate
3. Download detailed results as CSV

### Analytics
1. View match distribution
2. Analyze TGV profiles (radar charts)
3. Identify strengths and gaps
4. Compare multiple candidates

## Case Study Context
This is Step 3 of a Data Analyst case study focusing on:
- Success pattern discovery (Step 1)
- SQL matching logic (Step 2)
- AI-powered dashboard (Step 3) ← You are here

## License
Internal use only - Company X
