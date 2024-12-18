import csv

from pinecone_related.init import get_pinecone_index
from utils.helpers import convert_to_unix_timestamp


def query_pinecone_with_timerange(pinecone_index, namespace, start_date=None, end_date=None, host_url=None):
    """
    Query pinecone with date range
    :param pinecone_index: Pinecone index name
    :param namespace: Pinecone namespace
    :param start_date: Start date in YYYY-MM-DD format (optional)
    :param end_date: End date in YYYY-MM-DD format (optional)
    :param host_url: Organization ID (optional)
    :return: List of matches
    """
    index = get_pinecone_index(pinecone_index)
    query_filter = {}
    # Check if 'start_date' date is provided and add to filter
    if start_date:
        start_date_unix_timestamp = convert_to_unix_timestamp(start_date)
        query_filter.setdefault('created_at', {})['$gt'] = start_date_unix_timestamp

    # Check if 'end_date' date is provided and add to filter
    if end_date:
        end_date_unix_timestamp = convert_to_unix_timestamp(end_date)
        query_filter.setdefault('created_at', {})['$lt'] = end_date_unix_timestamp

    ## Add host_url filter
    if host_url:
        query_filter["host_url"] = host_url
    
    # Fetch documents
    response = index.query(
        vector=[0]*1536,  
        filter=query_filter,
        top_k=10, 
        namespace=namespace,
        include_metadata=True,
    )

    return response["matches"]

# print(query_pinecone_with_timerange("drmalpani", "main", "2024-01-24", "2024-01-26", "1"))