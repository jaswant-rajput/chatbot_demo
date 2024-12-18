import re
import time
import traceback
from datetime import datetime
from constants.misc import ERROR_LOG_PATH, INFO_LOG_PATH
import requests
from redis import Redis
from rq import Queue
from loguru import logger
import settings


# For logging into file
logger.add(
    ERROR_LOG_PATH,
    rotation="00:00",
    format="{time:HH:mm:ss} | {message}",
    level="ERROR",
    backtrace=False,
    diagnose=True,
    enqueue=True
)
logger.add(
    INFO_LOG_PATH,
    rotation="00:00",
    format="{time:HH:mm:ss} | {message}",
    level="INFO",
    enqueue=True
)

redis_conn = None
logging_queue = None
if settings.DEBUG != True:
    redis_conn = Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
    logging_queue = Queue(connection=redis_conn, name="logging_queue")


def log(message:str, error_message=None)->None:
    if error_message is None:
        logger.info(message)
    if error_message is not None:
        logger.exception(message)
        if (settings.DEBUG != True):
            add_task_to_logging_queue(save_error_log, message, error_message)

def add_task_to_logging_queue(task, *args, **kwargs):
    """
    Enqueue a task with a variable number of arguments and named keyword arguments.

    :param task: The task function to be executed.
    :param *args: A variable number of arguments to be passed to the task.
    :param **kwargs: A variable number of named keyword arguments to be passed to the task.
    :return: The result of the queue's enqueue method, typically a job object.
    """
    try:
        if logging_queue is None:
            log("ERROR: Invalid queue. Starting a thread instead")
            return start_background_thread(task, *args, **kwargs)
        job = logging_queue.enqueue(task, *args, **kwargs, description="")
        log(F"Started Logging Job {job.id}")
        return job.id
    except Exception as e:
        log("Error in add_task_to_logging_queue starting background thread", traceback.format_exc())
        try:
            return start_background_thread(task, *args, **kwargs)
        except Exception as e:
            log("Error in add_task_to_logging_queue in starting thread", traceback.format_exc())


def is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def get_vector_id(vector):
    """
        This function is used to obtain value of 'vec_id' from the input vector.

        Args:
            vector (dict): A dictionary representing a vector which is expected to contain a 'vec_id' key.

        Returns:
            str or None: The value associated with the 'vec_id' key if it exists, otherwise None.
    """

    try:
        return int(vector.get("id", -1))
    except KeyError:
        return None


def generate_final_prompt(messages: list, prompt, prompt_context=None):
    """
        Generates the final prompt to be used. It combines the system's prompt with additional context and the conversation messages.

        Args:
            messages (list): A list of dictionaries representing the messages exchanged between the user and the system.

            prompt (str): The main prompt message to be presented to the user.

            prompt_context (str, optional): Additional context to be included in the prompt, if any.
        
        Returns:
            list: A list of dictionaries representing the final prompt. Each dictionary contains
            'role' (indicating whether the message is from the system or the user) and 'content'
            (the message content) keys.
    """
    
    try:
        content = prompt if prompt_context is None else prompt + "\n\nCONTEXT:" + prompt_context
        final_prompt = [{
            "role": "system",
            "content": content
        }]

        for message in messages:
            # sender = message.get('sender', 'USER')
            # message_text = message.get('message', 'No Message')
            # agent = "assistant" if sender is "AI" else "user"
            final_prompt.append({
                "role": "assistant" if message.get('sender', 'USER') == "AI" else "user",
                "content": message.get('message', 'No Message')
            })
        return final_prompt
    except Exception as e:
        log("Error in generate_final_prompt", traceback.format_exc())
        return None


def start_background_thread(function, *args):
    """
        This function checks if the provided function is callable and then starts a new thread to execute 
        the function with the supplied arguments.

        Args:
            function (callable): The function to be executed in the background thread.
            *args: The arguments to be passed to the function.

        Returns:
            None
    """
    try:
        if not callable(function):
            log(f"Function {function} is not callable")
            return
        import threading
        threading.Thread(target=lambda: function(*args)).start()
    except Exception as e:
        log(f"Error in start_background_thread for Function {function}", traceback.format_exc())


def remove_hashtags(message: str) -> str:
    """
    Remove hashtags from a message, except those that are part of a URL.
    """
    try:
        # Pattern to identify hashtags that are not part of URLs
        pattern = r'(?<![\w/.-])(#\w+)(?![\w.-])'

        # Remove hashtags that match the pattern
        text_without_hashtags = re.sub(pattern, '', message)

        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r'\s+', ' ', text_without_hashtags)

        return cleaned_text.strip()
    except Exception as e:
        log("Error in remove_hashtags", traceback.format_exc())
        return message


def convert_to_int(output):
    if (isinstance(output, (int, float))):
        return int(output)
    if (isinstance(output, (complex))):
        return int(output.real)
    match = re.search(r'\d+', output)
    if match is not None:
        return int(eval(match.group()))
    return 1


# Find and format plain URLs
def replace_url(match):
    url = match.group(1)
    return f"[Read More]({url})"


def add_message_source_to_g(key, value):
    """
    
        Add a message source to the Flask application context's.
        
        Args:
            key (str): Identifier for the message to be stored.
            value: The value of the message source to be stored.

        Returns:
            None
    """

    try:
        from flask import g
        if 'sources' not in g:
            g.sources = {}
        g.sources[key] = value
    except:
        # Handle the error here
        log(f"Error in add_message_source_to_g", traceback.format_exc())


# Find and format plain emails
def replace_email(match):
    email = match.group(1)
    return f"[{email}](mailto:{email})"


def format_links_and_emails_as_markdown(text):
    """
        This function searches the input text for URLs and email addresses and converts them 
        to Markdown link format if they are not already formatted as such. URLs are converted 
        to "[Read More](URL)" and email addresses are converted to "[email](mailto:email)".

        Args:
            text (str): The input text containing URLs and email addresses.

        Returns:
            str: The text with URLs and email addresses formatted as Markdown links.
    """
    try:
        # This regex matches links that are already in Markdown format
        markdown_links = re.findall(r"\[.*?\]\(.*?\)", text)
        # This regex finds all URLs
        urls = re.findall(r"(https?://[^\s]+)", text)

        # This regex finds all email addresses
        emails = re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text
        )

        # We convert each URL to markdown only if it is not already in markdown
        for url in urls:
            markdown_version = f"[Read More]({url})"
            if markdown_version not in markdown_links:
                text = re.sub(
                    r"\b" + re.escape(url) + r"\b(?!.*?\))", markdown_version, text
                )

        # We convert each email to markdown in a similar fashion
        for email in emails:
            markdown_email = f"[{email}](mailto:{email})"
            if markdown_email not in markdown_links:
                text = re.sub(r"\b" + re.escape(email) + r"\b", markdown_email, text)

        return text
    except:
        log("Error in format_links_and_emails_as_markdown", traceback.format_exc())
        return text


def split_sentences(message: str) -> list:
    """
    Split a message into sentences using the PySBD sentence tokenizer.

    Args:
    message (str): The input message to be split into sentences.

    Returns:
    list: A list of sentences.
    """
    log(["message in split_sentence", message])
    if message is None or message == "":
        return message
    try:
        from pysbd import Segmenter
        segmenter = Segmenter(language='en', clean=False)

        return segmenter.segment(message)
    except ImportError:
        log("PySBD not installed. Please install PySBD to use this function.")
        return [message]
    except Exception as e:
        log("Error in split_sentences", traceback.format_exc())
        return [message]


def save_error_log(subject, error_log, platform="Flask"):
    try:
        log("SAVING ERROR LOG")
        requests.post(f"{settings.LARAVEL_BASEURL}/save_error_log",
                      json={"subject": subject, "error_log": str(error_log), "platform": platform})
    except Exception as e:
        log(traceback.format_exc())
        return False


def save_message_data_to_db(message_id, values):
    try:
        log("SAVING MESSAGE DATA TO DB")
        for x in values:
            if x is None:
                continue
            #     TODO: Don't call API 5 times back-to-back. Using for loop here is very inefficient.
            try:
                response = requests.post(f"{settings.LARAVEL_BASEURL}/save_flask_log",
                                         json={"message_id": message_id, "data": str(x)})
                response.raise_for_status()
            except Exception as e:
                log("Error in save_message_data", traceback.format_exc())

    except Exception as e:
        log("Error in save_message_data", traceback.format_exc())
        return False


def process_messages(messages):
    """
    Formats the received messages by changing them from a list of dictionaries to a list of strings.

    Args:
        messages (list): A list of dictionaries representing messages. Each dictionary
                         should contain keys 'sender' and 'message'.
        namespace (str): A string representing the namespace.

    Returns:
        tuple: A tuple containing the namespace, a list of processed messages, and the
               current message.
               - The processed messages list contains strings formatted as "AI: <message>"
                 if the sender is 'AI', and "USER: <message>" otherwise.
               - The current message is the last message in the processed messages list.
    """
    if len(messages) == 0:
        return []
    messages_array = []
    for message in messages:
        if message.get('sender') == "AI":
            # Remove Markdown links from AI answer, 
            markdown_link = r"\[(.+)\]\(.+\)"
            ai_message = re.sub(markdown_link, '', message.get('message'))
            message['message'] = ai_message
            messages_array.append({**message, "message":ai_message})
        else:
            messages_array.append({**message})

    return messages_array


# Define a function to check if a text is a URL
def is_url(text):
    try:
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'|(www\.[a-z0-9\.-]+)'  # www.domain
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
            r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return re.match(regex, text) is not None
    except Exception as e:
        log("Error in is_url", traceback.format_exc())
        return False


def get_url_from_vectors(relevant_sections):
    """
        This function sorts the provided list of relevant sections based on their scores in descending order.
        It then retrieves the 'read_more_link' and 'read_more_label' from the highest-scoring section, ensuring the URL is valid. If the URL is missing a label, a default label of "Read More" is applied.

        Args:
            relevant_sections (list): A list of dictionaries, each representing a relevant section.

        Returns:
            tuple: A tuple containing:
                - url (str): The URL from the highest-scoring relevant section or an empty string if no valid URL is found.
                - url_label (str): The label associated with the URL or an empty string if no valid URL is found.
    """
    try:
        if len(relevant_sections) > 0:
            relevant_sections = sorted(
                    relevant_sections,
                    key=lambda v: -v.get('score', 0),
                )
            url = relevant_sections[0].get('read_more_link', '')
            url_label = relevant_sections[0].get('read_more_label', '')

            if url != '' and is_url(url):
                if len(url_label) == 0:
                    url_label = "Read More"
                log(f"Link added: {url}")
                log("Relevant Section for URL generation")
                log(relevant_sections[0] if len(relevant_sections) > 0 else "No relevant section")
                return url, url_label
    except Exception as e:
        log("Error in get_url_from_vectors", traceback.format_exc())
        pass

    return "", ""


def convert_to_unix_timestamp(date_str):
    """
    Convert a date string in the format 'YYYY-MM-DD' to a Unix timestamp.

    Args:
    date_str (str): Date string in the format 'YYYY-MM-DD'.

    Returns:
    int: Unix timestamp representing the given date.
    """
    try:
        # Parse the date string to a datetime object
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # Convert the datetime object to a Unix timestamp
        unix_timestamp = int(time.mktime(date_obj.timetuple()))
        return unix_timestamp
    except ValueError as e:
        raise ValueError("Invalid date format. Please use 'YYYY-MM-DD'.") from e


def convert_to_standard_date(unix_timestamp):
    """
    Convert a Unix timestamp to a date string in the format 'YYYY-MM-DD'.

    Args:
    unix_timestamp (int): Unix timestamp.

    Returns:
    str: Date string representing the given Unix timestamp.
    """
    try:
        # Convert the Unix timestamp to a datetime object
        date_obj = datetime.fromtimestamp(unix_timestamp)
        # Format the datetime object to a date string
        print(date_obj.strftime("%Y-%m-%d"))
        return date_obj.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError("Invalid Unix timestamp.") from e


def extract_values_from_request(data):
    """
        Extracts values from received body data.

        Args:
            data (dict): Received data.
        
        Returns:
            list: A list of extracted values.
    """
    messages = data.get('messages')
    message_id = data.get('message_id')
    host_url = data.get('host_url')
    filters = data.get('filters', {})
    org_id = data.get('org_id')
    prompt = data.get('prompt')
    pinecone_index = data.get('pinecone_index')
    namespace = data.get('namespace')
    closure_msg = data.get('closure_msg')
    unsure_msg = data.get('unsure_msg')
    sender_country = data.get('sender_country')
    org_description = data.get('org_description')
    sender_city = data.get('sender_city')
    conversation_status = data.get('conversation_status')
    buckets = data.get('buckets')

    return messages, message_id, host_url, filters, org_id, prompt, pinecone_index, namespace, closure_msg, unsure_msg, sender_country, sender_city, conversation_status, buckets, org_description


# load json safely
def load_json(json_str):
    try:
        from json import loads
        return loads(json_str)
    except Exception as e:
        log("Error in load_json", traceback.format_exc())
        return None


# def get_bucket_ids(buckets):
#     return [x['id'] for x in buckets]

def process_buckets(buckets):
    try:
        buckets_str = ""
        for x in buckets:
            buckets_str += f"ID: {x['id']}, NAME: {x.get('name')}, DESCRIPTION: {x.get('description')}\n"
        return buckets_str
    except Exception as e:
        raise e
#################### UNUSED FUNCTIONS ###########################

# def preprocess_tweets(tweets):
#     def filter_non_conversational_tweets(tweets):
#         return [tweet for tweet in tweets if
#                 not tweet.startswith("RT ") and "http" not in tweet and not tweet.startswith("@")]

#     def remove_short_irrelevant_tweets(tweets, min_tweet_length=10):
#         return [tweet for tweet in tweets if len(tweet.split()) >= min_tweet_length]

#     def handle_mentions_hashtags(tweet):
#         tweet = re.sub(r"@\w+", "", tweet)  # Remove mentions
#         tweet = re.sub(r"#\w+", "", tweet)  # Remove hashtags
#         return tweet

#     filtered_tweets = filter_non_conversational_tweets(tweets)
#     relevant_tweets = remove_short_irrelevant_tweets(filtered_tweets)
#     cleaned_tweets = [handle_mentions_hashtags(tweet) for tweet in relevant_tweets]
#     return cleaned_tweets


# def init_email_config(app):
#     app.config['MAIL_SERVER'] = settings.MailConfig.smtp
#     app.config['MAIL_PORT'] = settings.MailConfig.port
#     app.config['MAIL_USE_TLS'] = settings.MailConfig.use_tls
#     app.config['MAIL_USE_SSL'] = settings.MailConfig.use_ssl
#     app.config['MAIL_USERNAME'] = settings.MailConfig.username
#     app.config['MAIL_PASSWORD'] = settings.MailConfig.password

#     mail = Mail(app)

#     return mail


# def send_email(email_address, app):
#     try:
#         mail = init_email_config(app)
#         # Create a message
#         msg = Message(subject='Hello, this is a test email', recipients=[email_address])
#         msg.body = 'This is the body of the email'

#         # Send the message
#         mail.send(msg)

#     except Exception as e:
#         log(traceback.print_exc())
