import os

import numpy as np
from dotenv import load_dotenv
from flask import request, jsonify
from langchain import OpenAI
from scipy.spatial.distance import cosine

from ml_models.gpt_helpers import create_embedding

load_dotenv()
client = OpenAI(api_key=os.environ.get('OPENAI_KEY'), organization=os.environ.get('OPENAI_ORG'))


def is_unanswered(text):
    """
    Evaluates the response text to determine if the text answered the user's query properly.

    Args:
    text (str): The text to be evaluated.

    Returns:
    int: 0 if the text did not have information the user was asking for, 1 if the text answered the user query properly.
    """

    # Define trigger phrases that indicate a lack of specific information or an inability to provide it
    trigger_phrases = [
        "I'm sorry, but I can't provide",
        "I'm sorry, but I don't have",
        "I don't have that information",
        "I can't provide the information you're looking for",
        "I'm unable to assist with that",
        "the provided information does not mention",
        "the information provided in the context does not mention"
    ]

    # Check if the first sentence of the text contains any of the trigger phrases
    first_sentence = text.split('.')[0]
    for phrase in trigger_phrases:
        if phrase in first_sentence:
            return 1

    # If none of the trigger phrases are found in the first sentence, assume the query was answered
    return 0


def get_top_negative_score_questions(qna_list, limit=10):
    """Return the top 10 negative score questions."""
    negative_score_questions = []

    for idx, question in enumerate(qna_list):
        qna_list[idx]["score"] = 0

        if question.get("unanswered"):
            qna_list[idx]["score"] -= 2.5

        if isinstance(question.get('answer_vote'), int) and question.get('answer_vote') < 0:
            qna_list[idx]["score"] -= 1

    return sorted(qna_list, key=lambda x: x["score"], reverse=False)[:limit]


def convert_to_qna_chucks(req_data):
    arr = []
    for i, entry in enumerate(req_data):
        if entry["sender_id"] != 1 and entry["message"].count(" ") > 2:
            output = {
                "question": entry["message"],
                "unanswered": entry["unanswered"],
                "message_id": entry['id'],
                "created_at": entry['created_at'],
                "chat_id": entry["chat_id"],
            }
            if i + 1 < len(req_data) and req_data[i + 1]["chat_id"] == entry["chat_id"] and req_data[i + 1][
                "sender_id"] == 1:
                output.update({
                    "answer_vote": req_data[i + 1]["vote"],
                })

                if entry["unanswered"] == 0:
                    output["unanswered"] = is_unanswered(req_data[i + 1]["message"])
            arr.append(output)
    return arr


def get_top_questions_by_times_asked(qna_list, top_n=10):
    # qna_list = [item for item in qna_list if item["question"] != "" and item.get("answer") != []]
    return sorted(qna_list, key=lambda x: x["times_asked"], reverse=True)[:top_n]


def generate_unique_questions(qna_list):
    import pandas as pd
    df = pd.DataFrame(qna_list)
    df['ada_embedding'] = df["question"].apply(lambda x: create_embedding(x))
    df['identifier'] = np.nan
    df.reset_index(inplace=True)

    def search_question_in_dataframe(row):
        question = row['question']
        embedding = create_embedding(question)
        # TODO: Compare embedding with df['ada_embedding'] and get second most matching row
        df['similarity_score'] = df['ada_embedding'].apply(lambda x: compare_cosine_similarity(embedding, x))

        top_matches = df.sort_values('similarity_score', ascending=False).head(2)

        if top_matches.iloc[1]["similarity_score"] >= 0.88:
            if not pd.isna(df.at[top_matches.index[0], 'identifier']):
                df.at[top_matches.index[1], 'identifier'] = df['identifier'].iloc[top_matches.index[0]]
            elif not pd.isna(df.at[top_matches.index[1], 'identifier']):
                df.at[top_matches.index[0], 'identifier'] = df['identifier'].iloc[top_matches.index[1]]
            else:
                random_num = np.random.rand()
                df.at[top_matches.index[0], 'identifier'] = random_num
                df.at[top_matches.index[1], 'identifier'] = random_num
        else:
            random_num = np.random.rand()
            df.at[top_matches.index[0], 'identifier'] = random_num

        if row["index"] not in [top_matches.index[0], top_matches.index[1]]:
            df.at[row["index"], 'identifier'] = df['identifier'].iloc[top_matches.index[0]]

    df.apply(search_question_in_dataframe, axis=1)

    df.drop(columns=["ada_embedding", "similarity_score"])

    unique_identifier_dicts = []
    seen_identifiers = set()

    for d in df.to_dict(orient='records'):
        identifier = d['identifier']
        if identifier not in seen_identifiers:
            d['times_asked'] = 1
            unique_identifier_dicts.append(d)
            seen_identifiers.add(identifier)
        else:
            for index, unique_d in enumerate(unique_identifier_dicts):
                if unique_d.get("identifier") == identifier:
                    unique_identifier_dicts[index]["times_asked"] += 1
                    break

    return unique_identifier_dicts


def compare_cosine_similarity(answer_embed, context_embed):
    return 1 - cosine(answer_embed, context_embed)


def analysis_main():
    req_data = request.get_json()
    qna_list = convert_to_qna_chucks(req_data['messages'])

    merged_msgs = generate_unique_questions(qna_list)

    # merged_messages = merge_similar_questions_using_nlp(qna_list)  # more accurate
    top_messages = get_top_questions_by_times_asked(merged_msgs, 10)

    ## Aggregating the data
    aggregated_data = get_top_negative_score_questions(merged_msgs, 10)

    return jsonify({"top_unanswered": aggregated_data, "top_questions": top_messages})
