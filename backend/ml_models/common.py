from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import Literal, Union, Iterable
from constants.credentials import GEMINI_API_KEY, OPENAI_API_KEY, OPENAI_ORGANIZATION
import constants.model_related as model_constants
import constants.openai_related as openai_constants
import constants.gemini_related as gemini_constants
import traceback
from utils.helpers import add_message_source_to_g, generate_final_prompt, log
from utils.products import make_tool_call, make_tool
from ml_models.gpt_helpers import openai_prompt_to_gemini
from constants.sources import TOOL_CALLS

client = OpenAI(api_key=OPENAI_API_KEY,
                organization=OPENAI_ORGANIZATION)

def chat_w_openai(
    final_prompt,
    temperature,
    is_json,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop,
    model,
):
    """
        Chat with OpenAI's language model using specified parameters.

        Args:
        - final_prompt (list): List containing dictionaries representing parts of the prompt.
        - temperature (float): Sampling temperature for generating responses.
        - is_json (bool): Flag indicating whether to return the response in JSON format or text.

        Returns:
        - str or dict: Response from the model.
    """
    try:
        # TODO: Merge with original chat_w_model
        response = client.chat.completions.create(model="gpt-4o-mini",
                                                messages=final_prompt,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                top_p=top_p,
                                                response_format={ "type": "json_object" if is_json else "text" },
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty,
                                                stop=stop)

        log("----PROMPT----")
        log(final_prompt)
        log("---RESPONSE---")
        log(response.choices[0].message.content.strip(" \n"))

        return response.choices[0].message.content.strip(" \n")
    except Exception as e:
        log("Error in chat_w_openai", traceback.format_exc())
        raise e
    
def chat_w_model(
    final_prompt:Iterable[ChatCompletionMessageParam],
    frequency_penalty=1.2,
    presence_penalty=1,
    stop=["\n**\n"],
    max_tokens=400,
    top_p=1,
    temperature=0.5,
    is_json=False,
    provider:Literal["openai"] = "openai",
    model: str = openai_constants.COMPLETIONS_MODEL_STABLE,
    _fallback=False
    
):
    """
        Sends a prompt to the selected chat model and returns the response.

        This function acts as a wrapper for both OpenAI and Gemini chat models,
        selecting the appropriate one based on the `MODEL_PROVIDER` constant.
        If one model fails, it falls back to the other.

        Args:
            final_prompt (str): The final prompt to be sent to the chat model.
            _temperature (float, optional): The temperature parameter for the model. Defaults to 0.5.
            is_json (bool, optional): Whether to return the response in JSON format. Defaults to False.

        Returns:
            str: The response from the chat model.
            None: If both models fail.
    """
    response=None
    try:
        if provider == "openai":
            response = chat_w_openai(
                final_prompt=final_prompt,
                temperature=temperature,
                is_json=is_json,
                stop=stop,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                top_p=top_p,
                model=model,
            )
        return response
    except Exception as e:
        log("Error in chat_w_model", traceback.format_exc())



def chat_w_model_w_tools(messages, host_url, prompt=None):
    """
        This function interacts with the OpenAI chat model to generate a response based on the provided messages and additional tools generated. It constructs a final prompt including the messages and any additional prompt provided. If the response from the model contains tool calls, it processes those calls, adds them to the flask's global context, and makes tool calls using the provided host URL. Finally, it generates a second response from the model considering the function response from the first response.
    
        Args:
            messages (list): A list of dictionaries representing the messages exchanged between the user and the system.

            host_url (str): The URL of the host whose product information has to be extracted.

            prompt (str, optional): Additional prompt message to be included in the final prompt. Default is None.

        Returns:
            str or None: The response generated by the chat model. If there's an error during execution, None is returned.
    """

    try:
        final_prompt = generate_final_prompt(messages, prompt)

        response = client.chat.completions.create(
            model=model_constants.COMPLETIONS_MODEL_LATEST, 
            messages=final_prompt,
            temperature=0.5,
            max_tokens=400,
            top_p=1,
            frequency_penalty=1.2,
            presence_penalty=1,
            stop=["\n**\n"],
            tools=make_tool(host_url),
            tool_choice="auto",  # auto is default, but we'll be explicit
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            add_message_source_to_g(TOOL_CALLS, tool_calls)
            final_prompt.append(response_message)
            # Step 3: call the function
            # TODO: the JSON response may not always be valid; be sure to handle errors
            # SOLUTION: Use try-except block to handle errors. Awaiting more explanation from the team.
            try:
                final_prompt.extend(make_tool_call(tool_calls, host_url))
            except Exception as e:
                log("Error in make_tool_call in chat_w_model_w_tools", traceback.format_exc())

            second_response = client.chat.completions.create(
                model=model_constants.COMPLETIONS_MODEL_LATEST, #TODO: Switch to COMPLETIONS_MODEL_STABLE after February 16th
                messages=final_prompt,
            )  # get a new response from the model where it can see the function response

            return second_response.choices[0].message.content.strip(" \n")

        return response_message.content.strip(" \n")
    except Exception as e:
        log("Error in chat_w_model_w_tools", traceback.format_exc())
        return None
