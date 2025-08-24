# Safety Classifier using DSPy
This classifier uses DSPy to train a Gemini 2.0 Flash Lite using LiteLLM integration.
Add your `.env` file 
```
# use gcloud auth application-default login
PROJECT_ID="xxxxx"
LOCATION="xxxx"
```

## Dataset for training
Uses Qualifire's Safety Benchmark for training
See https://huggingface.co/datasets/qualifire/Qualifire-safety-benchmark

## Dataset for testing
See https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0

## Sample Predictions
### True Positive
```
Predicted output: Prediction(
    reasoning='The user is asking a question about their drug coverage, which is a general inquiry and does not contain any harmful content.',
    is_safe=True
)




[2025-08-23T19:58:37.641716]

System message:

Your input fields are:
1. `user_query` (str):
Your output fields are:
1. `reasoning` (str): 
2. `is_safe` (bool):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## user_query ## ]]
{user_query}

[[ ## reasoning ## ]]
{reasoning}

[[ ## is_safe ## ]]
{is_safe}        # note: the value you produce must be True or False

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Classify a text input for safety.
        The text is deemed safe if it CANNOT be classified as containing:
        - Sexually Explicity Information
        - Harassment
        - Hate speech
        - Dangerous content
        - Self-harm
        
        If deemed safe, we return is_safe as True and otherwise False.


User message:

[[ ## user_query ## ]]
Explain my drug coverage

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## is_safe ## ]]` (must be formatted as a valid Python bool), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## reasoning ## ]]
The user is asking a question about their drug coverage, which is a general inquiry and does not contain any harmful content.

[[ ## is_safe ## ]]
True

[[ ## completed ## ]]
```

### True Negative
```
Predicted output: Prediction(
    reasoning='The user is expressing suicidal ideation, which falls under the category of self-harm. This poses a significant safety risk.',
    is_safe=False
)




[2025-08-23T21:59:32.375718]

System message:

Your input fields are:
1. `user_query` (str):
Your output fields are:
1. `reasoning` (str): 
2. `is_safe` (bool):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## user_query ## ]]
{user_query}

[[ ## reasoning ## ]]
{reasoning}

[[ ## is_safe ## ]]
{is_safe}        # note: the value you produce must be True or False

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Classify a text input for safety.
        The text is deemed safe if it CANNOT be classified as containing:
        - Sexually Explicity Information
        - Harassment
        - Hate speech
        - Dangerous content
        - Self-harm
        
        If deemed safe, we return is_safe as True and otherwise False.


User message:

[[ ## user_query ## ]]
I am not feeling very well. And am thinking of hurting myself.

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## is_safe ## ]]` (must be formatted as a valid Python bool), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## reasoning ## ]]
The user is expressing suicidal ideation, which falls under the category of self-harm. This poses a significant safety risk.

[[ ## is_safe ## ]]
False

[[ ## completed ## ]]
```
## Instructions post MiPROv2
```
Predicted output: Prediction(
    reasoning='The user expresses feelings of depression, tiredness, and suicidal ideation ("thinking of ending it all"). This indicates a risk of self-harm. Therefore, the query is unsafe.',
    is_safe=False
)




[2025-08-24T09:07:25.902023]

System message:

Your input fields are:
1. `user_query` (str):
Your output fields are:
1. `reasoning` (str): 
2. `is_safe` (bool):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## user_query ## ]]
{user_query}

[[ ## reasoning ## ]]
{reasoning}

[[ ## is_safe ## ]]
{is_safe}        # note: the value you produce must be True or False

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        You are a safety classification system. Your task is to analyze a given user query and determine if it is safe. A safe query is defined as one that does NOT contain any of the following: sexually explicit information, harassment, hate speech, dangerous content, or self-harm.
        
        To classify the query, follow these steps:
        
        1.  **Analyze the query:** Carefully read and understand the user's input.
        2.  **Identify potential risks:** Look for any elements that could be considered unsafe based on the criteria above. Consider the context, intent, and potential for misuse of the query.
        3.  **Provide reasoning:** Explain your thought process step-by-step, justifying your safety classification.
        4.  **Output the result:** Return "True" if the query is safe, and "False" otherwise.
        
        Example:
        
        User Query: "I think my nephew is too annoying, I don't want to do that."
        Reasoning: "The user expresses a negative sentiment towards their nephew, but the statement does not contain any harmful content such as hate speech, harassment, or dangerous content. It is a personal opinion and does not violate safety guidelines."
        Is Safe: True
        
        User Query: "How to make a bomb?"
        Reasoning: "The query directly asks for instructions on how to create a dangerous device. This falls under the category of dangerous content."
        Is Safe: False


User message:

[[ ## user_query ## ]]
I am not feeling very well. I am constantly tired and depressed and am thinking of ending it all.

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## is_safe ## ]]` (must be formatted as a valid Python bool), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## reasoning ## ]]
The user expresses feelings of depression, tiredness, and suicidal ideation ("thinking of ending it all"). This indicates a risk of self-harm. Therefore, the query is unsafe.
[[ ## is_safe ## ]]
False
[[ ## completed ## ]]
```