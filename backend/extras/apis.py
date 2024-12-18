import json
import traceback

from flask import request, jsonify

from constants.model_related import SMALL_TALK, ANSWERING_QUESTION, SHARING_FEEDBACK, END_CONVERSATION, \
    PRODUCT_RELATED_QUERY, intent_map
from constants.sources import INTENT, STANDALONE_QUESTION, VECTOR_IDS
from ml_models.user_facing import estimate_intent, make_standalone_question
from pinecone_related.query_pinecone import fetch_prompt_context_array
from utils.helpers import log, process_messages, start_background_thread
from utils.log_functions import save_sources_log


def convert_to_pinecone_structure(read_more_link="", source_url="", score="", vector_id="", text="", ):
    return {"vector_id": vector_id, "metadata": {
        "score": score, "text": text, "read_more_link": read_more_link, "source_url": source_url,
    }}


def fetch_vectors_from_conversation():
    try:
        messages = request.args.get('messages')
        messages = json.loads(messages)
        host_url = request.args.get('host_url')
        pinecone_index = request.args.get('pinecone_index')
        message_id = request.args.get('message_id')
        namespace = request.args.get('namespace')

        messages = process_messages(messages)
        formatted_messages = [(f"{message.get('sender')}: {message.get('message')}") for message in messages]
        conversation = "\n".join(formatted_messages)
        if namespace is None:
            namespace = ''

        intent = estimate_intent(conversation)
        if intent in [SMALL_TALK, ANSWERING_QUESTION, SHARING_FEEDBACK]:
            return jsonify({'status': 200, 'message': "Successfully fetched vectors", 'data': [
                convert_to_pinecone_structure(text="Generic AI message, unrelated to data points")]})

            # TODO: Handle Product Related Query
        if intent == PRODUCT_RELATED_QUERY and host_url == "multibhasi":
            return jsonify({'status': 200, 'message': "Successfully fetched vectors", 'data': [
                convert_to_pinecone_structure(
                    text="Answer fetched from product details, ability to view this data is coming soon!")]})

        elif intent == END_CONVERSATION:
            return jsonify({'status': 200, 'message': "Successfully fetched vectors", 'data': [
                convert_to_pinecone_structure(text="Default AI message for ending the conversation")]})

        standalone_question = make_standalone_question(formatted_messages, formatted_messages[-1])
        chosen_sections = fetch_prompt_context_array(standalone_question, pinecone_index, namespace, host_url)
        formatted_chosen_sections = []

        # save logs to the database
        start_background_thread(save_sources_log, message_id, {
            INTENT: intent_map[intent],
            STANDALONE_QUESTION: standalone_question,
            VECTOR_IDS: [section["id"] for section in chosen_sections] if chosen_sections else [],
        })
        for chosen_section in chosen_sections:
            formatted_chosen_sections.append(
                convert_to_pinecone_structure(chosen_section["read_more_link"], chosen_section["source_url"],
                                              chosen_section["score"], chosen_section["id"],
                                              chosen_section["text"], ))
        return jsonify({'status': 200, 'message': "Successfully fetched vectors", 'data': formatted_chosen_sections})
    except Exception as e:
        log("Error in fetch_vectors_from_conversation", traceback.format_exc())
        return jsonify({'status': 500, 'message': "Failed to fetched vectors", 'data': str(e)})
