#!/usr/bin/env python
"""
Simple terminal loop for quick ad-hoc tests.
"""

from agent_core import agent

def main():
    print("Mental-Health Assistant (type 'quit' to exit)\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"quit", "exit"}:
            break
        print("Assistant:", agent(msg), "\n")

if __name__ == "__main__":
    main()
