# nodes/check_confidence_node.py

class ConfidenceCheckNode:
    def __init__(self, threshold=75.0):
        self.threshold = threshold

    def __call__(self, state):
        # ✅ Use dot notation for dataclass input
        confidence = state.confidence

        # Check confidence threshold
        if confidence < self.threshold:
            print(f"⚠️  Low confidence ({confidence}%). Asking for user confirmation.")
            new_label = input(f"🤔 Is the prediction \"{state.label}\" correct? (y/n): ").strip().lower()
            if new_label == "n":
                corrected = input("✅ Please enter the correct label (positive/negative): ").strip().lower()
                return {
                    "label": corrected,
                    "confidence": confidence,
                    "needs_fallback": True,
                    "corrected_by_user": True
                }

        return {
            "confidence": confidence,
            "needs_fallback": False
        }
