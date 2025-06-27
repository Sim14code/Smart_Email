# %%
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from datetime import datetime, timedelta
import urllib.parse
import json
import warnings

# %% [markdown]
# ### Step 1: Set WatsonX credentials and project info

# Optional: Suppress all warnings
warnings.filterwarnings("ignore")

# %%
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key="gJVkYUmdyFqErmxjsr4jFTyoTyKrE6lzCHjX8zCd00ve"
)

client = APIClient(credentials)

params = {
    "decoding_method": "greedy",
    "max_new_tokens": 200
}

model_id = "ibm/granite-3-2b-instruct"
project_id = "943be453-c93d-4692-a82e-6b77c2a41503"

model = ModelInference(
    model_id=model_id,
    api_client=client,
    params=params,
    project_id=project_id
)

# %% [markdown]
# ### Step 2: Sample email input

# %%
email_body = """
Hi, I couldn't make the meeting today. Can we reschedule it for Monday, July 1st at 3:00 PM?
Thanks, Alex
"""

# %% [markdown]
# ### Step 3: Define prompt to extract meeting information

# %%
messages = [
    {
        "role": "system",
        "content": "You are an orchestration agent. Extract meeting details from the email. Reply ONLY in JSON format with keys: title, date (YYYY-MM-DD), time (HH:MM), description."
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

# %% [markdown]
# ### Step 4: Get model response

# %%
response = model.chat(messages=messages)
print("Raw WatsonX Response:\n", response)

# %% [markdown]
# ### Step 5: Extract structured meeting info

# %%
try:
    extracted = json.loads(response["choices"][0]["message"]["content"])
    print("\n📌 Extracted Meeting Info:\n", json.dumps(extracted, indent=2))
except Exception as e:
    print("❌ Failed to parse JSON:", e)
    extracted = None

# %% [markdown]
# ### Step 6: Generate Google Calendar Link

# %%
def generate_calendar_link(title, date_str, time_str, description, duration_minutes=60):
    try:
        start_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        start_fmt = start_dt.strftime("%Y%m%dT%H%M00Z")
        end_fmt = end_dt.strftime("%Y%m%dT%H%M00Z")

        params = {
            "action": "TEMPLATE",
            "text": title,
            "dates": f"{start_fmt}/{end_fmt}",
            "details": description,
            "location": "Online"
        }

        base_url = "https://calendar.google.com/calendar/render"
        return f"{base_url}?{urllib.parse.urlencode(params)}"
    except Exception as e:
        return f"❌ Error generating link: {e}"

# %% [markdown]
# ### Step 7: Print Calendar Invite Link

# %%
if extracted:
    calendar_link = generate_calendar_link(
        title=extracted["title"],
        date_str=extracted["date"],
        time_str=extracted["time"],
        description=extracted["description"]
    )
    print("\n📅 Google Calendar Link:")
    print(calendar_link)
else:
    print("⚠️ No meeting details to generate calendar link.")
