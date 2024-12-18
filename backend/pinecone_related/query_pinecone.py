import traceback

from openai import OpenAI
from constants.misc import CHATGPT_MAX_TOKENS, SEPARATOR, MAX_TOKEN_BUFFER
import constants.credentials as creds
from ml_models.gpt_helpers import create_embedding
from pinecone_related.init import check_index_exists

client = OpenAI(api_key=creds.OPENAI_API_KEY,
                organization=creds.OPENAI_ORGANIZATION)
from pinecone_related.init import get_pinecone_index
from utils.helpers import add_message_source_to_g, log
from constants.sources import TOTAL_TOKENS_FETCHED, TOTAL_TOKENS_USED, VECTOR_IDS, VECTORS_INFO

# TODO: What if we get no results because of buckets
def query_from_pinecone(pinecone_index, namespace, query="Who are you?", _top_k=50, host_url=None, filters=None, buckets=[]):
    """
        This function retrieves a Pinecone index and generates an embedding for the input query.
        It applies optional metadata filters and sorts the results by relevance and optionally by bucket priority.

        Args:
            pinecone_index (str): The name of the Pinecone index to query.
            namespace (str): The namespace within the Pinecone index to search within.
            query (str, optional): The input query text to generate an embedding for. Default is "Who are you?".
            _top_k (int, optional): The number of top relevant document sections to retrieve. Default is 50.
            host_url (str): The URL of the organization user is querying about.
            filters (dict, optional): Additional metadata filters to apply to the query. Default is None.
            buckets (list, optional): A list of buckets to filter results by bucket ID and sort by bucket priority. Default is an empty list.

        Returns:
            list: A list of the most relevant document sections, sorted by score or by bucket priority and score.
    """
    try:
        pinecone_index = get_pinecone_index(pinecone_index)
        xq = create_embedding(query)
        metadata_filter = {} 
        most_relevant_document_sections = []
        # if filters is not None and isinstance(filters, dict):
        #     metadata_filter = filters
        # if host_url is not None:
        #     metadata_filter["host_url"] = host_url
        # if len(buckets) > 0 and isinstance(buckets, list):
        #     metadata_filter["bucket_id"] = {"$in":[bucket['id'] for bucket in buckets]}

        fetched_vectors = pinecone_index.query(
            vector=xq,
            top_k=_top_k,
            filter=metadata_filter,
            include_metadata=True,
            include_values=False,
            namespace=namespace
          )
        calculate_total_tokens_fetched(fetched_vectors["matches"])
        most_relevant_document_sections = [v for v in fetched_vectors["matches"] if v['score'] >= 0.7]

        if len(buckets) > 0 and isinstance(buckets, list):
            try:
                log("--SORTING BY: BUCKET PRIORITY, SCORE")
                bucket_priorities = {bucket['id']: bucket.get('priority', 1) for bucket in buckets}
                # log(bucket_priorities)
                # log(sorted(
                #     map(lambda x:{"bucket_id":x['metadata']['bucket_id'], "score":x["score"], "id":x["id"]}, most_relevant_document_sections),
                #     key=lambda v: (bucket_priorities[int(v['bucket_id'])], -v['score']),
                # ))
                return sorted(
                    most_relevant_document_sections,
                    key=lambda v: (bucket_priorities[int(v['metadata']['bucket_id'])], -v['score']),
                )
            except Exception as e:
                log("Error in sorting by bucket priority", traceback.format_exc())  

        log("--SORTING BY: SCORE")
        return sorted(most_relevant_document_sections, key=lambda doc: doc['score'], reverse=True)
    except Exception as e:
        log("Error in query_from_pinecone", traceback.format_exc())
        return []
    
def extract_section_values(section_index):
    try:
        """
        Extracts values from a section index with error handling.

        Parameters:
        - section_index: The section index dictionary from which to extract values.

        Returns:
        - A dictionary with extracted values and a boolean indicating if the max tokens limit was exceeded.
        """

        # Initialize default values
        score = 0
        vec_id = ""
        read_more_link = ""
        read_more_label = ""
        source_url = ""
        bucket_id = None
        action_id = None

        # Extract other values with error handling
        score = section_index.get('score', 0)
        vec_id = section_index.get('id', "")
        meta = section_index.get('metadata', {})

        if "metadata" in section_index:
            read_more_link = meta.get('read_more_link', "")
            source_url = meta.get('source_url', "")
            bucket_id = meta.get('bucket_id', None)
            action_id = meta.get('action_id', None)
            read_more_label = meta.get('read_more_label', "")

        # Return extracted values in a dictionary and whether max tokens limit was exceeded
        return {
            "score": score,
            "vec_id": vec_id,
            "read_more_link": read_more_link,
            "source_url": source_url,
            "bucket_id": bucket_id,
            "action_id": action_id, 
            "read_more_label": read_more_label
        }
    except Exception as e:
        log("Error in extract_section_values", traceback.format_exc())
        return {
            "score": 0,
            "vec_id": "",
            "read_more_link": "",
            "source_url": "",
            "read_more_label": ""
        }


def fetch_prompt_context_array(question, pinecone_index, namespace, host_url=None, filters={}, unsure_msg="I don't know", buckets=[]):
    """
        This function queries a Pinecone index to find the most relevant document sections based on the provided question.
        It processes the results, ensuring that the total token count does not exceed the maximum allowed by the ChatGPT model.
        The selected sections are returned with their metadata for use in generating responses.

        Args:
            question (str): The input question to generate an embedding for querying the Pinecone index.
            pinecone_index (str): The name of the Pinecone index to query.
            namespace (str): The namespace within the Pinecone index to search within.
            host_url (str): The URL of the organization user is querying about.
            filters (dict, optional): Additional metadata filters to apply to the query. Default is an empty dictionary.
            unsure_msg (str, optional): A fallback message to use if no relevant sections are found. Default is "I don't know".
            buckets (list, optional): A list of buckets to filter results by bucket ID and sort by bucket priority. Default is an empty list.

        Returns:
            list: A list of dictionaries containing the relevant document sections and their metadata. Each dictionary includes:
                - "score" (float): The relevance score of the section.
                - "vec_id" (str): The ID of the vector in the Pinecone index.
                - "read_more_link" (str): A link to read more about the section content.
                - "source_url" (str): The source URL of the section content.
                - "bucket_id" (int, optional): The bucket ID if applicable.
                - "read_more_label" (str): A label for the read more link.
                - "description" (str): The content of the section, prefixed with a separator and formatted.
                - "tokens_count" (int): The number of tokens in the section content.

            If no relevant sections are found, returns a list with a single dictionary containing the unsure message.
    """
    if host_url is None:
        return ""
    most_relevant_document_sections = query_from_pinecone(pinecone_index, namespace, question, host_url=host_url, filters=filters, buckets=buckets)

    chosen_sections = []
    chosen_sections_len = 0
    vector_ids = []
    vector_info = []
    for _, section_index in enumerate(most_relevant_document_sections):
        metadata:dict = section_index.get("metadata", {})
        tokens_count = metadata.get("tokens", 0)
        vector_id = int(metadata.get("id", section_index.get('id', -1)))
        #  Check if adding this section would exceed the max token limit
        if (chosen_sections_len + tokens_count) > CHATGPT_MAX_TOKENS:
            continue
        chosen_sections_len += tokens_count
        try:
            vector_ids.append(vector_id)
            vector_info.append({  
                "vector_id": vector_id,
                "bucket": next((bucket.get('title', None) for bucket in buckets if
                                int(bucket.get("id", -1)) == int(metadata.get("bucket_id", -2))),
                                "Bucket Not Used"),
                "score": section_index.get("score", None),
                "tokens_count": metadata.get("tokens", None)
              })
        except:
            log("Error in fetch_prompt_context_array (appending vector id and info)", traceback.format_exc())
        # TODO: If metadata doesn't have text / empty text, save it in vector_id list and fetch it from DB.
        chosen_sections.append({**metadata,"score": section_index.get("score", 0), "id": vector_id})
        if (CHATGPT_MAX_TOKENS - chosen_sections_len) < MAX_TOKEN_BUFFER:
            break

    # log the number of tokens used
    log(f"--total_tokens_used: {chosen_sections_len}")
    add_message_source_to_g(TOTAL_TOKENS_USED, chosen_sections_len)
    add_message_source_to_g(VECTOR_IDS, vector_ids)
    add_message_source_to_g(VECTORS_INFO, vector_info)
    if len(chosen_sections) == 0:
        return [{"read_more_link": "", "score": "", "id": -1,
                "text": SEPARATOR + unsure_msg
            }]

    return chosen_sections

def calculate_total_tokens_fetched(most_relevant_document_sections):
    """
        This function iterates over a list of document sections, extracts the token count from each section's metadata,
        sums up these token counts, logs the total, and adds the total to a global message source.

        Args:
            most_relevant_document_sections (list): A list of document sections, each represented as a dictionary.
                Each section dictionary should have a 'metadata' key containing a 'tokens' entry.

        Returns:
            None
    """
    try:
        total_tokens_fetched = sum([int(section_index.get('metadata',{}).get('tokens', 0)) for section_index in most_relevant_document_sections])
        log(f"--total_tokens_fetched: {total_tokens_fetched}")
        add_message_source_to_g(TOTAL_TOKENS_FETCHED, total_tokens_fetched)
    except:
        log("Error in calculate_total_tokens_fetched", traceback.format_exc())

def fetch_prompt_context(question, pinecone_index, namespace, host_url=None, filters=None, unsure_msg="", buckets=[]):
    """
        This function checks if the specified Pinecone index exists, then fetches the most relevant document sections
        for the input question. It processes the sections to ensure they do not exceed the maximum token limit and
        returns a concatenated string of the section descriptions along with their metadata.

        Args:
            question (str): The input question to generate an embedding for querying the Pinecone index.
            pinecone_index (str): The name of the Pinecone index to query.
            namespace (str): The namespace within the Pinecone index to search within.
            host_url (str, optional): A URL to filter results by host URL. Default is None.
            filters (dict, optional): Additional metadata filters to apply to the query. Default is None.
            unsure_msg (str, optional): A fallback message to use if no relevant sections are found. Default is an empty string.
            buckets (list, optional): A list of buckets to filter results by bucket ID and sort by bucket priority. Default is an empty list.

        Returns:
            tuple: A tuple containing:
                - str: A concatenated string of the descriptions of the chosen sections, formatted and prefixed with a separator.
                - list: A list of dictionaries containing the relevant document sections and their metadata. Each dictionary includes:
                    - "score" (float): The relevance score of the section.
                    - "vec_id" (str): The ID of the vector in the Pinecone index.
                    - "read_more_link" (str): A link to read more about the section content.
                    - "source_url" (str): The source URL of the section content.
                    - "bucket_id" (int, optional): The bucket ID if applicable.
                    - "read_more_label" (str): A label for the read more link.
                    - "description" (str): The content of the section, prefixed with a separator and formatted.
                    - "tokens_count" (int): The number of tokens in the section content.
    """
    try:
        if not check_index_exists(pinecone_index):
            raise Exception(f"Index {pinecone_index} does not exist")            
            
        chosen_sections = fetch_prompt_context_array(question, pinecone_index, namespace, host_url, filters, unsure_msg, buckets)
        
        return chosen_sections
    except Exception as e:
        log("Error in fetch_prompt_context", traceback.format_exc())
        raise e


# def query_pinecone_with_buckets(pinecone_index, namespace, question, host_url, filters, bucket_ids, _top_k=5):
#     most_relevant_document_sections = []
#     for bucket_id in bucket_ids:
#         filters["bucket_id"] = bucket_id
#         fetched_vectors =  query_from_pinecone(pinecone_index, namespace, question, host_url=host_url, filters=filters)
#         for x in fetched_vectors:
#             if x['score'] > 0.7:
#                 most_relevant_document_sections.append(x)
#                 if len(most_relevant_document_sections) >= _top_k:
#                     break
#     return most_relevant_document_sections


### OLD HYBRID QUERY METHOD
# # TODO: Optimize, remove redundancy
# def query_from_pinecone(pinecone_index, namespace, query="Who are you?", _top_k=10, host_url=None, filters=None, bucket_ids=None):
#     try:
#         pinecone_index = get_pinecone_index(pinecone_index)
#         xq = create_embedding(query)
#         metadata_filter = {} 
#         most_relevant_document_sections = []
#         if filters is not None and isinstance(filters, dict):
#             metadata_filter = filters
#         if host_url is not None:
#             metadata_filter["host_url"] = host_url

#         if bucket_ids is not None and isinstance(bucket_ids, list):
#             log("---Query Method: Bucket Query")
#             for bucket_id in bucket_ids:
#                 metadata_filter["bucket_id"] = bucket_id
#                 fetched_vectors = pinecone_index.query(vector=[xq], top_k=_top_k,
#                                         filter=metadata_filter,
#                                         include_metadata=True, namespace=namespace)
#                 for x in fetched_vectors["matches"]:
#                     if x['score'] > 0.7:
#                         most_relevant_document_sections.append(x)
#                         if len(most_relevant_document_sections) >= _top_k:
#                             break
#                 log(f"Got {len(fetched_vectors['matches'])} Vectors in Bucket {bucket_id}")
#             if len(most_relevant_document_sections) >= _top_k:
#                 calculate_total_tokens_fetched(most_relevant_document_sections)
#                 return sorted(most_relevant_document_sections, key=lambda doc: doc['score'], reverse=True)
#             else: 
#                 log("Query did not return enough results. Switching to normal query method.")
#                 metadata_filter.pop("bucket_id", None)
#                 most_relevant_document_sections = []
                
#         log("---Query Method: Normal Query")
#         res = pinecone_index.query(vector=[xq], top_k=_top_k,
#                                         filter=metadata_filter,
#                                 include_metadata=True, namespace=namespace)
#         for match in res["matches"]:
#             if match['score'] > 0.7:
#                 most_relevant_document_sections.append(match)
#                 if len(most_relevant_document_sections) >= _top_k:
#                     break
#         calculate_total_tokens_fetched(most_relevant_document_sections)
#         return sorted(most_relevant_document_sections, key=lambda doc: doc['score'], reverse=True)
#     except Exception as e:
#         log("Error in query_from_pinecone", traceback.format_exc())
#         return []