# nodes/fallback_node.py

class FallbackNode:
    def __init__(self):
        self.valid_labels = ["Positive", "Negative"]

    def __call__(self, context: dict) -> dict:
        print(f"\n⚠️  Low confidence ({context['confidence']}%).")
        print(f"🤖 Predicted: {context['label']}")
        print("🧠 Please help me out: What do you think this should be?")
        print("Options: [Positive / Negative]")

        while True:
            user_input = input("Your label → ").strip().capitalize()
            if user_input in self.valid_labels:
                context["label"] = user_input
                context["corrected_by_user"] = True
                break
            else:
                print("❌ Invalid input. Please enter either 'Positive' or 'Negative'.")

        return context
