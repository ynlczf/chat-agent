import os
import time
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

DB_URI = os.getenv("DB_URI", "postgresql://postgres:root@127.0.0.1:5432/testdb")

print("Connecting to database...")
start = time.time()

# Use context manager correctly
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    print(f"Connected in {time.time() - start:.2f}s")
    
    # Test list with None config
    print("Listing checkpoints (config=None, limit=10)...")
    start = time.time()
    try:
        items = list(checkpointer.list(None, limit=10))
        print(f"Found {len(items)} checkpoints in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test list with a specific thread_id
    print("\nListing checkpoints (with thread_id filter)...")
    start = time.time()
    try:
        items = list(checkpointer.list({"configurable": {"thread_id": "test-1"}}))
        print(f"Found {len(items)} checkpoints in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")
