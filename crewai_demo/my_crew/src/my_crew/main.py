from crew import MyCrew

def main():
    print("🚀 Starting CrewAI demo...\n")
    inputs = {"topic": "Generative AI"}
    result = MyCrew().crew().kickoff(inputs=inputs)
    print("\n✅ Crew finished successfully!")
    print(result.raw)

if __name__ == "__main__":
    main()
