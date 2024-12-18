import re

from openai import OpenAI

import constants.credentials as creds
client = OpenAI(api_key=creds.OPENAI_API_KEY,
                organization=creds.OPENAI_ORGANIZATION)

import settings
from ml_models.common import chat_w_model
from utils.helpers import log, convert_to_int


def prompt_chaining(question, context):
    content = " ".join(context)

    final_prompt = [{
        "role": "user",
        "content": f'Context: {content}\n\nQuestion: {question}\n\nDetermine whether Context is sufficient to answer the Question and repond with a JSON with the following format:\n' + '{"context_contains_answer": boolean, "justification": string}\nONLY respond with the JSON in the correct format.'
    }]
    return chat_w_model(final_prompt)


def refine_intent_estimator(user_input):
    """
    Refined version of the intent estimator to better distinguish between small talk, seeking information, and ending the conversation.
    """
    # TODO: Test this and if better, then use this
    user_input = user_input.lower()

    # Refined patterns
    small_talk_patterns = [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\byo\b", r"\bhow's it going\b", r"\bhow are you\b",
                           r"\bwhat's up\b", r"\bfeeling .+", r"you are .+", r"are you .+"]
    seeking_info_patterns = [r"\bwhat\b", r"\bhow\b", r"\bwhy\b", r"\bwhere\b", r"\bwhen\b", r"\bwhich\b", r"\bwho\b",
                             r"\bcan\b", r"\bcould\b", r"\bshould\b", r"\bwould\b", r"\bdo\b", r"\bdoes\b", r"\bdid\b",
                             r"\bhave\b", r"\bhas\b", r"\bhad\b", r"\bshare\b", r"\bneed\b", r"\bwant\b", r"\bi have\b",
                             r"\bi'm looking for\b", r"\binterested in\b"]
    ending_conversation_patterns = [r"\bok\b", r"\bokay\b", r"\bthanks\b", r"\bthank you\b", r"\bbye\b", r"\bsee you\b",
                                    r"\bgoodbye\b", r"\btalk later\b", r"\bcatch you later\b", r"\bthat's all\b",
                                    r"\bthat's it\b", r"\bno more questions\b", r"\bi'll leave now\b"]

    # Check patterns
    if any(re.search(pattern, user_input) for pattern in small_talk_patterns):
        return 0
    elif any(re.search(pattern, user_input) for pattern in seeking_info_patterns):
        return 1
    elif any(re.search(pattern, user_input) for pattern in ending_conversation_patterns):
        return 2
    else:
        return 1  # Default to seeking information


def estimate_intent_neo(query):
    final_prompt = """Given the conversation below, where last message is from user.\n
        Respond with 0 if user is making small talk,
        1 if user wants to converse further or asks a question,
        2 if user wants to end the conversation.\n
        \n\nhi: 0\nWat is your address: 1\nok thanks: 2\nI want investment for my startup: 1\nyo bro: 0\nDoes sun rise in the east?: 1\nyou are stupid: 0\ni have very few eggs remaining: 1\nshare details: 1\nOk. I will consult Dr Malpani clinic for this: 2\n
        chat history:
    """

    final_prompt += "\n" + query + ":"

    response = client.completions.create(prompt=final_prompt,
    **settings.COMPLETIONS_API_INENT)

    output = response.choices[0].text.strip(" \n")
    log(f"ESTIMATING INTENT OF USER: {output}")

    return convert_to_int(output)
