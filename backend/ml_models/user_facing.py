import json
import traceback

from openai import OpenAI

import constants.credentials as creds
from constants.model_related import (INTENT_ESTIMATOR_PROMPT as PROMPT, intent_map, NON_PRODUCT_RELATED_QUERY)
from utils.helpers import add_message_source_to_g

client = OpenAI(api_key=creds.OPENAI_API_KEY,
                organization=creds.OPENAI_ORGANIZATION)

try:
    from ml_models.common import chat_w_model
except ImportError:
    pass
from pinecone_related.query_pinecone import fetch_prompt_context
from extras.citations import get_response_with_citations
from utils.helpers import (convert_to_int, generate_final_prompt, log)
from constants.sources import GPT_RESPONSE, STANDALONE_QUESTION
from constants.common import INTENT_PREDICTION_MODEL


def make_standalone_question(querylist, current_message):
    """
        Reformulates a follow-up message in a chat into a standalone message with enough context on its own.

        This function takes a list of previous messages (querylist) and the current follow-up message (current_message) from a chat conversation.
        It constructs a prompt to ask a language model to rephrase the follow-up message as a standalone message. 
        The function returns the reformulated standalone message, or the original message if reformulation is not successful.

        Parameters:
        querylist (list of str): The list of previous messages in the conversation.
        current_message (str): The follow-up message that needs to be reformulated.

        Returns:
        str: The reformulated standalone message or the original follow-up message if reformulation fails.
    """
    chat_history = "\n".join(
        querylist[:-1]) if len(querylist) > 1 else "No chat history available"

    final_prompt = [{
        "role": "user",
        "content": f"""Given the following conversation between a user and a sales assistant and a follow up message, rephrase the follow up message sent by the user as a standalone message that carries enough context on its own.
        Conversation History Begin
        {chat_history}
        Conversation History End
        Follow up message Begin
        {current_message}
        Follow up message End
        IMPORTANT NOTE:
        1) Respond with the following JSON format: {{"STANDALONE QUESTION": "the single standalone message", "justification": "reason why you chose this standalone message and why it fits the requirement" }}
        2) Do NOT answer the message, just reformulate it if needed and otherwise return it as is.\
        3) Correct any spelling errors while processing text.
    """
    }]
    response = chat_w_model(final_prompt, 0.2, is_json=True)
    try:
        loaded_response = json.loads(response)
        add_message_source_to_g(STANDALONE_QUESTION, loaded_response)
        final_question = loaded_response.get("STANDALONE QUESTION", None)
        log(f"---STANDALONE QUESTION: {final_question}")
        # log(f"---JUSTIFICATION FOR STANDALONE QUESTION: {loaded_response.get('justification', '')}")
        if final_question == None or final_question == "":
            log(f"---No standalone question generated. Using current message as standalone question: {current_message}")
            return current_message
        return final_question
    except:
        log("Error in getting standalone question from JSON, sending current message", traceback.format_exc())
        add_message_source_to_g(STANDALONE_QUESTION, current_message)
        return current_message


def estimate_intent(conversation, org_description=None):
    """
        Attempts to find out what the user wishes to achieve by the conversation or user's intent.

        Args:
            query (list of str): List of formatted messages representing conversation between user and the system.

            org_description (str, optional): Description of the organization that the user is currently talking to.
        
        Returns:
            int: An integer representing the estimated intent of the user query. The possible values and their meanings are:
            - 0: Small Talk
            - 1: Product-related Query
            - 2: Non-product-related Query
            - 3: Sharing Feedback
            - 4: Answering Question
            - 5: End Conversation
    """
    try:
        if org_description is not None:
            org_prompt = "Here are the details of the organization that user is currently talking to: \"\"\"" + org_description + ".\"\"\"\nUse this information and user's conversation to decide.\n"
        else:
            org_prompt = ""
        final_prompt = [{
            "role": "system",
            "content": org_prompt + PROMPT
        },
            {
                "role": "user",
                "content": f"conversation starts here\n\"\"\"{conversation}\"\"\"\nconversation ends here\n"
            }]
        
        response = chat_w_model(
            final_prompt=final_prompt,
            temperature=0.1,
            max_tokens=60,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"],
            model=INTENT_PREDICTION_MODEL
            )

        output = convert_to_int(
            response)
        log(f"ESTIMATING INTENT OF USER: {intent_map[output]}")
        return output
    except Exception as e:
        log("Error in estimating intent", traceback.format_exc())
        return NON_PRODUCT_RELATED_QUERY


def answer_query_generic(messages, prompt=""):
    """
        Generates a generic response to continue the conversation based on the given messages and prompt.

        This function is used to generate a response for small talk or generic conversations. It appends
        a specific instruction to the prompt, indicating that the response should only make small talk and
        avoid mentioning specific details. The function then generates a final prompt by combining the
        messages and the modified prompt, and passes it to the `chat_w_model` function to generate the response.

        Args:
            messages (list): A list of messages representing the conversation history.
            prompt (str, optional): An optional prompt to guide the response generation. Defaults to an empty string.

        Returns:
            str: The generated response for the given messages and prompt, or None if an error occurs.
    """
    try:
        prompt += "\nONLY MAKE SMALL TALK to continue the conversation. Avoid mentioning specific details like address, phone number, cost, etc."
        final_prompt = generate_final_prompt(messages, prompt)
        return chat_w_model(final_prompt)
    except Exception as e:
        log("Error in answer_query_generic", traceback.format_exc())
        return None

def answer_query_generic_ncert(messages, prompt=""):
    try:
        final_prompt = generate_final_prompt(messages, prompt)
        return chat_w_model(final_prompt)
    except Exception as e:
        log("Error in answer_query_generic", traceback.format_exc())
        return None


# Answer with RAG
def answer_query_with_context(messages: list,
                              conversation: str,
                              standalone_question: str,
                              pinecone_index: str,
                              filters: dict,
                              host_url: str,
                              prompt: str,
                              namespace: str,
                              sender_city=None,
                              sender_country=None,
                              unsure_msg="I don't know",
                              buckets: list = []
                              ):
    """
        This function processes a standalone question by querying a Pinecone index to find relevant document sections.
        It then uses the fetched context to generate a response using a language model, adding additional metadata 
        like sender city and country if provided.

        Args:
            messages (list): A list of message dictionaries forming the conversation history.
            conversation (str): Conversation between user and model.
            standalone_question (str): The standalone question to be answered.
            pinecone_index (str): The name of the Pinecone index to query for relevant context.
            filters (dict): Metadata filters to apply to the Pinecone query.
            host_url (str): The URL of the organization user is querying about.
            prompt (str): The base prompt to be used for generating the response.
            namespace (str): The namespace within the Pinecone index to search within.
            sender_city (str, optional): The city of the sender, to be included in the prompt. Default is None.
            sender_country (str, optional): The country of the sender, to be included in the prompt. Default is None.
            unsure_msg (str, optional): A fallback message if no relevant context is found. Default is "I don't know".
            buckets (list, optional): A list of buckets to filter results by bucket ID and sort by bucket priority. Default is an empty list.

        Returns:
            tuple: A tuple containing:
                - str: The response generated by the language model.
                - str: The concatenated string of the descriptions of the chosen sections.
                - bool: A boolean indicating whether the question was answered.
                - list: A list of the relevant document sections and their metadata.
    """
    try:
        if buckets and len(buckets) > 0:
            log(f"---initial_bucket_details: {buckets}")
            buckets = get_relevant_buckets(standalone_question, buckets)
            log(f"---selected_buckets: {buckets}")
        relevant_sections = fetch_prompt_context(
            standalone_question, pinecone_index, namespace,
            host_url, filters, unsure_msg, buckets)
        if sender_city is not None and sender_country is not None:
            prompt += f"\n\nSender City: {sender_city}\nSender Country: {sender_country}"
        try:
            refined_gpt_response, action_id = get_response_with_citations(
                prompt, standalone_question, messages,
                conversation, relevant_sections, unsure_msg
            )
        except Exception as e:
            log(f"Error in getting response from GPT-3", traceback.format_exc())
            # TODO: Generate Answers without citations here
            refined_gpt_response, action_id = get_response_with_citations(
                prompt, standalone_question, messages,
                conversation, relevant_sections, unsure_msg
            )
        return refined_gpt_response, relevant_sections, action_id
    except Exception as e:
        log(f"Error in answering query with context", traceback.format_exc())
        return None, None, None


def normalize_json_response(response):
    """
         This function takes a JSON response string, parses it into a Python dictionary, 
        extracts specific fields, and logs the formatted response. It adds the parsed response 
        to a global message source and returns the extracted fields.

        Args:
            response (str): The JSON response string to be normalized.

        Returns:
            tuple: A tuple containing:
                - str: The value of the "response" key from the parsed JSON.
                - bool: The value of the "is_answered" key from the parsed JSON, converted to a boolean.
    """
    try:
        formatted_response = json.loads(response)
        add_message_source_to_g(GPT_RESPONSE, formatted_response)
        log(f"---FORMATTED JSON RESPONSE: {formatted_response}")
        gpt_response = formatted_response.get("response")
        is_answered_value = formatted_response.get("is_answered", True)
        is_answered = (is_answered_value.lower() == 'true') if isinstance(
            is_answered_value, str) else bool(is_answered_value)
        return gpt_response, is_answered
    except Exception as e:
        raise e


def get_relevant_buckets(standalone_question, buckets):
    try:
        # TODO: Remove if len(buckets) > 1 else None
        return buckets if isinstance(buckets, list) and len(buckets) >= 0 else []
        # TODO: Confirm and enable if to use chatgpt or not
        # final_prompt = [{
        #     "role": "system",
        #     "content": "In the BUCKETS, Predict relevant bucket for the given STANDALONE QUESTION. You can select maximum 3 buckets.The response should be sorted in descending order of how relevant the bucket is to the STANDALONE QUESTION. Respond with the following JSON format: {\"buckets\": [\"bucket_id\"]}"
        # }, {
        #     "role": "user",
        #     "content": f"STANDALONE QUESTION: {standalone_question}"
        # }, {
        #     "role": "system",
        #     "content": f"BUCKETS: {process_buckets(buckets)}"
        # }]
        # response = chat_w_model(final_prompt, 0.5, is_json=True)
        # buckets = load_json(response).get("buckets", None)
        # add_message_source_to_g(BUCKETS_USED, buckets)
        # return buckets
    except Exception as e:
        log("Error in getting relevant buckets", traceback.format_exc())
        return None

def get_second_last_user_intent(messages):
    """
        Retrieves the intent of the second last message from the user.

        This function filters the provided list of messages to find those sent by the user. It then checks
        if there are at least two such messages. If so, it returns the intent of the second last message
        sent by the user. If there are fewer than two messages from the user, it returns None.

        Args:
            messages (list): A list of message dictionaries representing the conversation history.
                Each message dictionary is expected to have at least the following keys:
                - "sender": A string indicating the sender of the message (e.g., "USER", "AI").
                - "intent" (optional): A string representing the intent of the message.
        
        Returns:
            str or None: The intent of the second last message sent by the user if it exists, otherwise None.
    """
    try:
        # Filter messages where sender is "USER"
        user_messages = [msg for msg in messages if msg["sender"] == "USER"]
        
        # Check if there are at least two messages from the user
        if len(user_messages) < 2:
            return None
        
        # Return the intent of the second last "USER" message
        return user_messages[-2].get("intent", None)
    except:
        log("Error in get_second_last_user_intent", traceback.format_exc())
        return None


##################### UNUSED FUNCTIONS #####################
#     def make_standalone_question_legacy(querylist):
#     final_prompt = [{
#         "role": "user",
#         "content": f"""Given a chat history and the latest user question \
# which might reference the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is.
# Chat History: {querylist}
# Standalone Question: """
#     }]
#     response = chat_w_model(final_prompt, 0.2)

#     log(f"---STANDALONE QUESTION: {response}")

#     return response


# def estimate_intent_legacy(query):
#     final_prompt = """Auto-correct any spelling errors while processing text. Respond with 0 if user is making small talk, 1 if user wants to converse further, 2 if user wants to end the conversation.\n\nhi: 0\nWat is your address: 1\nok thanks: 2\nI want investment for my startup: 1\nyo bro: 0\nDoes sun rise in the east?: 1\nyou are stupid: 0\ni have very few eggs remaining: 1\nshare details: 1\nOk. I will consult Dr Malpani clinic for this: 2\n What is the Cost for IVF?: 1"""

#     final_prompt += "\n" + query + ":"

#     response = client.completions.create(prompt=final_prompt,
#                                          **settings.COMPLETIONS_API_INENT)

#     output = response.choices[0].text.strip(" \n")
#     log(f"ESTIMATING INTENT OF USER: {output}")

#     return convert_to_int(output)
