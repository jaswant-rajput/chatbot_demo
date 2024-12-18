from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone.exceptions import NotFoundException
from constants.credentials import OPENAI_API_KEY, OPENAI_ORGANIZATION, PINECONE_API_KEY
# from ml_models.gpt_helpers import create_embedding_for_list
import traceback
from utils.helpers import log

client = OpenAI(api_key=OPENAI_API_KEY,
                organization=OPENAI_ORGANIZATION)
pc_client = Pinecone(api_key=PINECONE_API_KEY)


# def create_sample_embedding():
#     return create_embedding_for_list("Sample document text goes here")

def check_index_exists(index_name):
    # Get the list of all existing indexes
    indexes = pc_client.list_indexes()
    try:
        for index in indexes:
            if index.get('name') == index_name:
                return True
        return False
    except Exception as e:
        log("Error in check_index_exists", traceback.format_exc())
        raise e



def get_pinecone_index(pinecone_index):
    """
        This function attempts to retrieve an existing Pinecone index specified by `pinecone_index`.
        If the index does not exist, it is created with dimensions based on a sample embedding.

        Args:
            pinecone_index (str): The name of the Pinecone index to retrieve or create.

        Returns:
            Pinecone.Index: The Pinecone index object.
    """
    try:
        # if pinecone_index not in pc_client.list_indexes():
    #     pc_client.create_index(pinecone_index, dimension=len(create_sample_embedding()))
        # TODO: use host name to create index (improves performance)
        index = pc_client.Index(pinecone_index)
        return index
    except NotFoundException as e:
        raise f"Index {pinecone_index} not found" 

