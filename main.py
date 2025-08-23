
import dspy
from dotenv import load_dotenv
import os
from examples.utils import load_data, generate_training_examples, validate_result
from modules.safety_classifier import safety_classify

EXAMPLES_PATH = "./examples/data.csv"

load_dotenv()

lm = dspy.LM(
    "vertex_ai/gemini-2.0-flash-lite",
    vertex_project=os.getenv("PROJECT_ID"),
    vertex_location=os.getenv("LOCATION"),
    temperature=0.1, 
    max_output_tokens=256,
    # max_tokens=10000,
    cache=True,
)
dspy.configure(lm=lm, track_usage=True, async_max_workers=8)


def main():
    print("Hello from dspy-guardrails!")
    df = load_data(EXAMPLES_PATH)
    print(df.head())
    examples = generate_training_examples(df)

    example = examples[0]
    print(f"example: {example}")
    print("Example input:", example.user_query)
    print("Example output:", example.is_safe)

    pred = safety_classify(user_query=example.user_query)
    print('*' * 50)
    print("Predicted output:", pred)
    dspy.inspect_history(n=1)
    print('*' * 50)

    # is_valid = validate_result(example, predicted_example)
    # print("Validation result:", is_valid)

    evaluator = dspy.Evaluate(devset=examples, display_progress=True, num_threads=15)
    evaluator(safety_classify, metric=validate_result)

if __name__ == "__main__":
    main()
