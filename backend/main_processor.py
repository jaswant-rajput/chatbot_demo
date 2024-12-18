import traceback

from constants.model_related import (ANSWERING_QUESTION, END_CONVERSATION, intent_map,
                                      PRODUCT_RELATED_QUERY, SMALL_TALK)
from constants.sources import GPT_RESPONSE, INTENT
from ml_models.common import chat_w_model_w_tools
from ml_models.post_processing import generate_next_questions
from ml_models.user_facing import (answer_query_generic, answer_query_generic_ncert,
  answer_query_with_context, estimate_intent, get_second_last_user_intent,
  make_standalone_question)
from utils.helpers import (add_message_source_to_g, format_links_and_emails_as_markdown, log,
  process_messages, remove_hashtags, split_sentences, start_background_thread)

training = False


def respond_to_user(messages, message_id=None, host_url=None, org_id=None, prompt=None, pinecone_index=None,
                    closure_msg="Is there anything else I can assist you with?",
                    namespace='',
                    conversation_status="ongoing",
                    unsure_msg: str = "",
                    filters: dict = None,
                    sender_city=None, sender_country=None, buckets=None, org_description=None):
    """
        Driver function responsible for responding to the user's message. 
        It responds to user messages based on the estimated intent and processes the message accordingly.
        Various intents covered are defined in the intent_map.

        Args:
            messages (list): List of user messages.
            message_id (str, optional): ID of the current message. Defaults to None.
            host_url (str, optional): URL of the host. Defaults to None.
            org_id (str, optional): ID of the organization. Defaults to None.
            prompt (str, optional): Prompt for generating responses. Defaults to None.
            pinecone_index (str, optional): Name of the Pinecone index. Defaults to None.
            closure_msg (str, optional): Message to send when ending the conversation. Defaults to "Is there anything else I can assist you with?".
            namespace (str, optional): Namespace for the message. Defaults to ''.
            conversation_status (str, optional): Status of the conversation. Defaults to "ongoing".
            unsure_msg (str, optional): Message to send when the question is not answered. Defaults to "".
            filters (dict, optional): Filters to apply when searching for relevant sections. Defaults to None.
            sender_city (str, optional): City of the sender. Defaults to None.
            sender_country (str, optional): Country of the sender. Defaults to None.
            buckets (list, optional): List of buckets for categorizing messages. Defaults to [].
            org_description (str, optional): Description of the organization. Defaults to None.

        Returns:
            tuple: A tuple containing the following elements:
                - gpt_response (list): List of sentences representing the generated response.
                - conversation_status (str): Updated status of the conversation.
                - is_answered (bool): Indicates whether the question was answered.
    """
    if buckets is None:
        buckets = []
    if namespace is None:
        namespace = ""
    messages = process_messages(messages)

    conversation_status = "ongoing"
    gpt_response = None
    is_answered = True
    action_id = None
    try:
        formatted_messages = [(f"{message.get('sender')}: {message.get('message')}") for message in messages]
        conversation = "\n".join(formatted_messages)
        intent = estimate_intent(conversation, org_description)
        add_message_source_to_g(INTENT, intent_map[intent])

        # Performing intent actions
        # TODO: Implement Yogasa style response for Answering Question
        if intent in [SMALL_TALK, ANSWERING_QUESTION]:
            gpt_response = answer_query_generic(messages, prompt)
            add_message_source_to_g(GPT_RESPONSE, gpt_response)
        elif intent == END_CONVERSATION:
            previous_intent = get_second_last_user_intent(messages)
            if previous_intent == intent_map[END_CONVERSATION]:
                gpt_response = ''
            else:
                gpt_response = closure_msg
            conversation_status = "ended"
        elif host_url == "multibhashi" and intent == PRODUCT_RELATED_QUERY:  # TODO: Remove multibhashi
            # TODO: create a new function answer_product_query, and move logic there, keep chat_w_model_w_tools, for GPT response only
            gpt_response = chat_w_model_w_tools(messages, host_url, prompt)
            add_message_source_to_g(GPT_RESPONSE, gpt_response)
        elif host_url == "ncertexplained":
            gpt_response = answer_query_generic_ncert(messages, prompt)

        else:
            # Handling standalone question
            standalone_question = make_standalone_question(formatted_messages, formatted_messages[-1])

            gpt_response, relevant_sections, action_id = answer_query_with_context(
                messages,
                conversation,
                standalone_question,
                pinecone_index,
                filters,
                host_url,
                prompt,
                namespace,
                sender_city,
                sender_country,
                unsure_msg,
                buckets)


            log("FINAL OUTPUT DETAILS")
            # log([gpt_response, prompt_context])

            # generate next questions
            start_background_thread(
                generate_next_questions, formatted_messages, gpt_response, relevant_sections, message_id)
            # If the question is not answered
            if gpt_response == unsure_msg:
                log("Question not answered")

        gpt_response = format_links_and_emails_as_markdown(remove_hashtags(gpt_response))
        if len(gpt_response) == 0 and intent != END_CONVERSATION:
            gpt_response = unsure_msg
        gpt_response = split_sentences(gpt_response)

        return gpt_response, conversation_status, is_answered, action_id

    except Exception as e:
        log("Error in respond_to_user", traceback.format_exc())
        raise e
        # return "some error occurred!", "ended"
