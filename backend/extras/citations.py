import traceback
import re

from ml_models.common import chat_w_model
from utils.helpers import (generate_final_prompt, log, add_message_source_to_g)
from constants.model_related import (CITATION_QA_TEMPLATE, CITATION_REFINE_TEMPLATE)
from constants.sources import (GPT_RESPONSE_REFINED, GPT_RESPONSE_WITH_CITATION, VECTORS_USED)


def get_response_with_citations(prompt:str, standalone_question:str, messages:list, conversation: str, relevant_sections:list, unsure_msg:str):
    """
        Retrieve a response with citations from relevant sections.

        Args:
            prompt (str): The initial prompt.
            standalone_question (str): A standalone question.
            messages (list): List of messages.
            conversation (str): Conversation between user and model.
            relevant_sections (list): List of relevant sections with 'id' and 'description'.

        Returns:
            tuple: A tuple containing the formatted response with citations,
                a flag indicating if the response is answered, and an action ID.
    """
    try:
        context_msg = ""
        for source in relevant_sections:
            context_msg += f"{source['id']}:\n{source['text']}\n\n"
        strict_prompt = prompt + "\n" + CITATION_QA_TEMPLATE.format(
            context_str = context_msg,
            unsure_msg=unsure_msg,
        )
        final_prompt = generate_final_prompt(messages, strict_prompt)
        response = chat_w_model(final_prompt, frequency_penalty=0)
        add_message_source_to_g(GPT_RESPONSE_WITH_CITATION, response)
        if response == unsure_msg:
            return unsure_msg, -1
        refined_strict_prompt = CITATION_REFINE_TEMPLATE.format(
            existing_answer = response,
            context_msg = context_msg,
            # conversation=conversation
        )
        final_prompt = [{
            "role": "user",
            "content": refined_strict_prompt
        }]
        refined_response = chat_w_model(final_prompt, temperature=1, presence_penalty=0, frequency_penalty=0)
        add_message_source_to_g(GPT_RESPONSE_REFINED, refined_response)
        formatted_response, action_id = replace_ids_with_links(refined_response, relevant_sections)
        return formatted_response, action_id
    except Exception as e:
            log(f"Error in get_response_with_citations", traceback.format_exc())
            raise e


def replace_ids_with_links(response, relevant_sections):
    """
        Process citations in the response, extract source vectors from relevant sections,
        format citations, and replace vector IDs with links.

        Args:
            response (str): The response containing vector IDs.
            relevant_sections (list): List of relevant sections with 'id', 'score', and optionally 'action_id' and 'read_more_link'.

        Returns:
            tuple: A tuple containing the formatted response with replaced links and an action ID.
    """
    try:
        square_bracket_pattern = r'\[(.*?)\]'
        # Converting [12, 23] -> [12] [23] etc.
        response = re.sub(square_bracket_pattern, lambda x: ' '.join([f"[{item.strip()}]" for item in x.group(1).split(',')]), response)

        # Avoid matching markdown texts
        markdown_pattern = r'\[([^\[\]]*)\](\((.*?)\))?'
        matches = re.findall(markdown_pattern, response)
        id_filter = [text for text, _, url in matches if url]

        int_inside_square_bracket_pattern = r'\[(\d+)\]'
        source_ids = re.findall(int_inside_square_bracket_pattern, response)
        source_ids = [id for id in source_ids if id not in id_filter]
        unique_source_ids = list(dict.fromkeys(source_ids))

        log(f"--matches in response: {source_ids}")
        add_message_source_to_g(VECTORS_USED, list(unique_source_ids))
        log(f"--total citations in response: {len(unique_source_ids)}")
        action_id = None
        seen_links = {}
        counter = 1
        for v_id in unique_source_ids:
            max_score = -1
            section = None
            try:
                section = next((relevant_section for relevant_section in relevant_sections if relevant_section.get("id", None) == int(v_id)), None)
            except:
                log("Error in replace_ids_with_links", traceback.format_exc())
            if section:
                link = section.get("read_more_link", "")
                if section.get('action_id', None):
                    if section['score'] > max_score:
                        max_score = section['score']
                        action_id = int(section['action_id'])
                if link:
                    if link in seen_links:
                        response = response.replace(f"[{v_id}]", f"{[[seen_links[link]]]}({link})")
                    else:
                        response = response.replace(f"[{v_id}]", f"{[[counter]]}({link})")
                        seen_links[link] = counter
                        counter += 1
                else:
                    response = response.replace(f"[{v_id}]", "")
            else:
                response = response.replace(f"[{v_id}]", "")
                
        return response, action_id

    except Exception as e:
        log(f"Error in replace_ids_with_links", traceback.format_exc())
        # TODO: remove all sources brackets
        return response, None