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
    "max_tokens": 100  # This is the accepted name
}

# Step 3: Select model and project
model_id = "ibm/granite-3-2b-instruct"  # Or use granite if preferred
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

# Step 5: Define your Intent Decoder email input
email_body = """
Hello,

I am unable to log in to my account even though I'm using the correct password. Can you please help me reset it?

Thanks,
Alex
"""

messages = [
    {
        "role": "system",
        "content": "You are an intent decoder. Extract the user's intent from the email body clearly and concisely."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": email_body
            }
        ]
    }
]

# Step 6: Get model response
response = model.chat(messages=messages)

# Step 7: Print the intent
print("Detected Intent:\n", response)


# %%
# Raw full response (optional for logging/debugging)
# print("Full response:\n", response)

# Cleaned output
clean_intent = response["choices"][0]["message"]["content"]
print("\n🔍 Decoded Customer Intent:")
print(clean_intent)


# %%



