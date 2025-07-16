# 1 set up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add keys

# 2 make vectors
python scripts/01_build_index.py

# 3 quick test
python scripts/02_run_chat_cli.py

# 4 web chat
streamlit run app/streamlit_app.py
