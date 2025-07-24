"""
Simple terminal loop for quick ad-hoc tests.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_core import agent

def main():
    """Run the mental health assistant."""
    print("Mental-Health Assistant (type 'quit' to exit)\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"quit", "exit"}:
            break
        print("Assistant:", agent(msg), "\n")

if __name__ == "__main__":
    main()
