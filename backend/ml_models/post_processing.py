import traceback

from openai import OpenAI

import constants.credentials as creds
client = OpenAI(api_key=creds.OPENAI_API_KEY,
                organization=creds.OPENAI_ORGANIZATION)
import requests
from scipy.spatial.distance import cosine

import settings
from ml_models.common import chat_w_model
from ml_models.gpt_helpers import create_embedding
from utils.helpers import log


def embedding_comparison(answer, context):
    content = " ".join(context)

    answer_embed = create_embedding(answer)
    context_embed = create_embedding(content)
    cosine_similarity = 1 - cosine(answer_embed, context_embed)
    # TODO: use semantic search
    log(f"Embed score : {cosine_similarity}")
    return cosine_similarity


def chatgpt_comparison(answer, context):
    content = " ".join(context)

    final_prompt = [{
        "role": "user",
        "content": f"Context: {content}\n\nAnswer: {answer}\n\nRespond with 'true' if Answer can be inferred from Context, else respond with 'false'. ONLY respond with 'true' or 'false'"
    }]
    response = chat_w_model(final_prompt, temperature=0.2)
    log(f"GPT Response : {response}")
    return response


def evaluate_answer(query, answer, context=None):
    final_prompt = """Give probability score between 0 to 1 of how likely it is that the question was answered 
    satisfactorily\nQ: In which direction does the Sun rise?\nA: Sun rises in the east.\nScore: 1\n\nQ: What's your 
    name?\nA: Hi! I am John Morris\nScore: 1\n\nQ: How can I pitch to you ?A: To pitch to Dr. Malpani, be concise and 
    clear about your idea's problem, solution, market potential, and why you're the right person for the job\nScore: 
    0.4\n\nQ: What is the Cost for IVF?\nA: The cost of IVF at Dr. Malpani's clinic may vary depending on individual circumstances and treatment requirements\nScore: 0.2"""

    final_prompt = "Below is a Question asked by a user and an answer by an AI. How satisfactorily did the AI answer the user? Answer with just a number score between 0 to 10."

    final_prompt += "\n\nQuestion: " + query + "\nAnswer: " + answer + "\nScore:"

    response = client.completions.create(prompt=final_prompt,
    **settings.COMPLETIONS_API_INENT)

    log("----PROMPT----")
    log(final_prompt)

    log("---RESPONSE---")
    log(response.choices[0].text.strip(" \n"))

    return response.choices[0].text.strip(" \n")

def generate_next_questions(prev_conversation, current_message, relevant_sections, message_id):
    if settings.DEBUG == True:
        log("Skipping generate_next_questions")
        return
    """
        This function constructs a conversation string from the previous conversation and the current message.
        It then sends this conversation along with additional context and a message ID to a specified endpoint
        to generate the next set of questions. The results are logged, and any errors encountered are also logged.

        Args:
            prev_conversation (str or list): The previous conversation. Can be a string or a list of strings.

            current_message (str): The current message from the AI.

            context (str): Additional context to be included in the request.
            
            message_id (str): The ID of the current message.

        Returns:
            None
    """
    try:
      if isinstance(prev_conversation, list):
          conversation = "\n".join(prev_conversation) + f"\nAI: {current_message}"
      else:
          conversation = prev_conversation + f"\nAI: {current_message}"
      context = "\n".join([section['text'] for section in relevant_sections])
      resp = requests.post(f"{settings.BACKGROUND_FLASK_ENDPOINT}/generate_next_questions", data={
                      "message_id": message_id,
                      "context": context,
                      "conversation": conversation                                             
                      })
      resp.raise_for_status()
      log(f"Generated next questions for message_id: {message_id}")
    except Exception as e:
          log(f"Error in generating next questions for message_id: {message_id}", traceback.format_exc())
