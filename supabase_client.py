import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
SUPABASE_CONFIGURED = bool(url and key)
supabase = create_client(url, key) if SUPABASE_CONFIGURED else None