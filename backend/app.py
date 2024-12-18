import json
import traceback

from dotenv import load_dotenv
from flask import Flask, jsonify, request, g
from flask_cors import CORS, cross_origin
from marshmallow import ValidationError

from constants.sources import CONVERSATION_STATUS, GPT_RESPONSE
from extras.apis import fetch_vectors_from_conversation
from main_processor import respond_to_user
from utils.helpers import (add_message_source_to_g, log, add_task_to_logging_queue, extract_values_from_request)
from utils.log_functions import save_sources_log
from utils.schemas import ChatRequestSchema

load_dotenv()
app = Flask(__name__)
app.secret_key = 'your_secret_key'
CORS(app)


@app.before_request
def log_request_info():
    try:
        log(f'Request: {request.method} {request.url} from {request.headers.get("X-Forwarded-For")} using {request.user_agent}')
    except:
        pass


@app.route('/')
def helloWorld():
    return "Flask App Running"


app.add_url_rule('/fetch_vectors_from_conversation',
                 view_func=fetch_vectors_from_conversation, methods=['GET'])


@cross_origin()
@app.route('/send_message', methods=['POST'])
def chat_with_gpt():
    try:
        schema = ChatRequestSchema()
        data = schema.load(request.json)
        # extracting values from the request
        messages, message_id, host_url, filters, org_id, prompt, pinecone_index, namespace, closure_msg, unsure_msg, sender_country, sender_city, conversation_status, buckets, org_description = extract_values_from_request(
            data)

        log(f"----SEND_MESSAGE_PARAM for {message_id}----")
        log(json.dumps(request.json, indent=4))

        if filters is not None and not isinstance(filters, dict):
            filters = None

        response, conversation_status, is_answered, action_id = respond_to_user(messages, message_id=message_id,
                                                                     host_url=host_url,
                                                                     org_id=org_id,
                                                                     filters=filters,
                                                                     pinecone_index=pinecone_index,
                                                                     prompt=prompt, closure_msg=closure_msg,
                                                                     namespace=namespace,
                                                                     conversation_status=conversation_status,
                                                                     unsure_msg=unsure_msg,
                                                                     sender_city=sender_city,
                                                                     sender_country=sender_country,
                                                                     buckets=buckets, org_description=org_description
                                                                     )
        # preparing the response json
        res_json = {
            "ai_response": response,
            'conversation_status': conversation_status,
        }
        if int(is_answered) == 0:
            res_json["unanswered"] = 1
        if action_id != -1:
            res_json['action_id'] = action_id
        final_json = jsonify(
            {'status': 200, 'message': 'Chat fetched successfully!', 'data': res_json})

        log(final_json.get_json())
        add_message_source_to_g(CONVERSATION_STATUS, conversation_status)
        add_message_source_to_g(GPT_RESPONSE, response)
        add_task_to_logging_queue(
            save_sources_log, message_id, g.get("sources", {}))
        return final_json

    except ValidationError as e:
        log("Error in send_message", traceback.format_exc())
        return jsonify({'status': 400, 'message': 'Missing required fields', 'data': e.messages}), 400

    except Exception as e:
        log("Error in send_message", traceback.format_exc())
        return jsonify({'status': 500, 'message': str(e), 'data': None})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
