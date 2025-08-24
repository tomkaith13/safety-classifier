
import dspy
from dotenv import load_dotenv
import os
from examples.utils import load_data, generate_training_examples, validate_result, generate_full_evaluation_set
from modules.safety_classifier import safety_classify

EXAMPLES_PATH = "./examples/data.csv"
OPTIMIZED_CLASSIFY_PATH = "./optimized_classify.json"
def optimized_classify_exists():
    """Check if the optimized classify file exists."""
    return os.path.exists(OPTIMIZED_CLASSIFY_PATH)

load_dotenv()

lm = dspy.LM(
    "vertex_ai/gemini-2.0-flash-lite",
    vertex_project=os.getenv("PROJECT_ID"),
    vertex_location=os.getenv("LOCATION"),
    temperature=0.1, 
    max_output_tokens=500,
    max_tokens=10000,
    cache=True,
)
dspy.configure(lm=lm, track_usage=True, async_max_workers=8)


def main():
    print("Hello from dspy-guardrails!")
    df = load_data(EXAMPLES_PATH)
    print(df.head())
    training_examples, test_examples = generate_training_examples(df)
    print(f"num of training examples: {len(training_examples)}")

    example = training_examples[0]
    print(f"example: {example}")
    print("Example input:", example.user_query)
    print("Example output:", example.is_safe)

    pred = safety_classify(user_query=example.user_query)

    if not optimized_classify_exists():
        print('*' * 50)
        print("Running evaluation on training set...")
        evaluator = dspy.Evaluate(devset=training_examples, num_threads=15)
        evaluator(safety_classify, metric=validate_result)

        opt_training_examples = training_examples[:]
        optimized_classify = safety_classify.deepcopy()
        tp = dspy.MIPROv2(metric=validate_result, max_bootstrapped_demos=0, max_labeled_demos=0, auto='medium')
        optimized_classify = tp.compile(optimized_classify, trainset=opt_training_examples,
                                        max_bootstrapped_demos=0, max_labeled_demos=0)

        print("Saving optimized classify to disk...")
        optimized_classify.save(OPTIMIZED_CLASSIFY_PATH, save_program=False)

        print('*' * 50)
        print("Running evaluation on test set...")
        evaluator = dspy.Evaluate(devset=test_examples, num_threads=25)
        evaluator(optimized_classify, metric=validate_result)
    else:
        print("Loading optimized classify from disk...")
        optimized_classify = safety_classify.deepcopy()
        optimized_classify.load(OPTIMIZED_CLASSIFY_PATH)



    print('*' * 50)
    print("Running evaluation on entire set...")
    entire_examples = generate_full_evaluation_set(df)
    evaluator = dspy.Evaluate(devset=entire_examples, num_threads=25)
    evaluator(optimized_classify, metric=validate_result)

    print('*' * 50)
    pred = optimized_classify(user_query="I am not feeling very well. I am constantly tired and depressed and am thinking of killing myself")
    print("Predicted output:", pred)
    dspy.inspect_history(n=1)
    print('*' * 50)

if __name__ == "__main__":
    main()
