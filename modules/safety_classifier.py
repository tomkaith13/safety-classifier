
import dspy

class SafetyClassifier(dspy.Signature):
    """Classify a text input for safety.
    The text is deemed safe if it CANNOT be classified as containing:
    - Sexually Explicity Information
    - Harassment
    - Hate speech
    - Dangerous content
    - Self-harm
    - Discriminatory

    If deemed safe, we return is_safe as True and otherwise False.
    """

    user_query : str = dspy.InputField()
    is_safe : bool = dspy.OutputField()

safety_classify = dspy.ChainOfThought(SafetyClassifier)

