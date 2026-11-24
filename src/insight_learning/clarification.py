"""Optional wording clarification through a configured Workers Free account."""
import json
import os
import re
from urllib.error import URLError
from urllib.request import Request, urlopen


def clarify(question, domain):
    account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    token = os.getenv("CLOUDFLARE_AI_TOKEN")
    if os.getenv("LEARNER_WORKERS_FREE_ENABLED") != "1" or not token or not re.fullmatch("[a-fA-F0-9]{32}", account):
        return None
    if len(question) > 500:
        return None
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/meta/llama-3.2-3b-instruct"
    instruction = (f"Rewrite this {domain} analytics question in clear English. Correct typos. "
                   "Preserve every value, filter, unit and requested operation. Return only the question.")
    body = {"messages": [{"role": "system", "content": instruction},
                         {"role": "user", "content": question}], "max_tokens": 128}
    request = Request(url, json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    try:
        with urlopen(request, timeout=8) as response:
            data = json.load(response)
        answer = data["result"]["response"].strip()
    except (URLError, TimeoutError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if not answer or len(answer) > 500 or re.findall(r"\d+(?:\.\d+)?", question) != re.findall(r"\d+(?:\.\d+)?", answer):
        return None
    return answer
