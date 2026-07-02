"""
Replays sample conversations against the local /chat endpoint
and prints the agent's replies turn by turn. Run this after starting
the server to quickly check if the agent is behaving correctly.
"""
import requests
import time

BASE_URL = "http://localhost:8000"

conversations = [
    {
        "name": "C1 - Senior leadership",
        "turns": [
            "We need a solution for senior leadership.",
            "The pool consists of CXOs, director-level positions; people with more than 15 years of experience.",
            "Selection — comparing candidates against a leadership benchmark.",
            "Perfect, that's what we need.",
        ]
    },
    {
        "name": "C2 - Senior Rust engineer",
        "turns": [
            "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?",
            "Yes, go ahead. Should I also add a cognitive test for this level?",
            "That works. Thanks.",
        ]
    },
    {
        "name": "C3 - AI Research Intern",
        "turns": [
            "I am hiring an AI research intern for our ML team.",
            "Fresh graduate, no experience required. They will work on NLP and computer vision projects.",
            "Yes add a personality test as well.",
            "What is the difference between the Global Skills Assessment and the AI Skills assessment?",
            "Perfect, that's what we need. Thanks.",
        ]
    },
]

for convo in conversations:
    print(f"\n{'='*60}")
    print(f"Testing: {convo['name']}")
    print('='*60)

    history = []

    for turn_text in convo["turns"]:
        history.append({"role": "user", "content": turn_text})

        try:
            response = requests.post(
                f"{BASE_URL}/chat",
                json={"messages": history},
                timeout=35
            )

            if response.status_code != 200:
                print(f"\nUser: {turn_text}")
                print(f"ERROR: Server returned {response.status_code}")
                print(f"Body: {response.text}")
                break

            data = response.json()

            print(f"\nUser: {turn_text}")
            print(f"Agent: {data['reply']}")
            print(f"Recommendations: {len(data['recommendations'])} items")
            if data['recommendations']:
                for r in data['recommendations']:
                    print(f"  - {r['name']} ({r['test_type']})")
            print(f"End of conversation: {data['end_of_conversation']}")

            history.append({"role": "assistant", "content": data["reply"]})

        except Exception as e:
            print(f"\nUser: {turn_text}")
            print(f"ERROR: {e}")
            break

        time.sleep(15)