# %%
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# %%


# Step 1: Set credentials
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",  # Watsonx endpoint (adjust if needed)
    api_key="gJVkYUmdyFqErmxjsr4jFTyoTyKrE6lzCHjX8zCd00ve"
)

client = APIClient(credentials)

# Step 2: Define inference parameters
params = {
    "decoding_method": "sample",
    "max_new_tokens": 600,
    "temperature": 0.7,  # Slightly creative, still focused
    "top_k": 50,
    "top_p": 0.9
}

# Step 3: Select model and project
model_id = "ibm/granite-3-8b-instruct"  
project_id = "943be453-c93d-4692-a82e-6b77c2a41503"
space_id = None
verify = False

# Step 4: Create model interface
model = ModelInference(
    model_id=model_id,
    api_client=client,
    params=params,
    project_id=project_id,
    space_id=space_id,
    verify=verify,
)

email_content = """
Hello,

I am not coming to the meerting tomorrow.

Thanks,
Alex
"""

prompt = f"""
 You are an AI assistant that helps users write professional, polite, and well-structured replies to emails.

    Below is an email from a user. Your task is to generate 3 different reply suggestions based on 3 possible interpretations of the situation. The replies should reflect:

    1 . A positive outcome 
    2. A neutral outcome 
    3. A negative or empathetic response or any gesture like thank you as per the context of email.

    Each reply should be concise, clear, and maintain a professional tone.

    EMAIL:
    \"\"\"
    {email_content}
    \"\"\"

    REPLY RECOMMENDATIONS:
    1.
    Reply:

    2. 
    Reply:

    3. 
    Reply:
"""


# Step 6: Get model response
response = model.generate(prompt=prompt)

# Step 7: Print the intent
print("Suggestion:\n", response)

try:
    replies = response['results'][0]['generated_text']
except (KeyError, IndexError):
    replies = "Failed to parse the model output."

# Clean display
print("\nReply Suggestions:\n")
print(replies)

