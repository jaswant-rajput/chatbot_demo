import settings

LARAVEL_BASEURL = settings.LARAVEL_BASEURL
endpoint_get_context = LARAVEL_BASEURL + '/get_data_from_vectors'
endpoint_get_latest_vector_id = LARAVEL_BASEURL + '/get_latest_vector_id'
endpoint_get_org_meta_data = LARAVEL_BASEURL + '/get_meta_data_of_org'
endpoint_save_next_questions = LARAVEL_BASEURL + '/save_next_questions'
endpoint_sync_vector_data = LARAVEL_BASEURL + '/sync_vector_data'

endpoint_save_error_log = LARAVEL_BASEURL + '/save_error_log'
endpoint_save_flask_log = LARAVEL_BASEURL + '/save_flask_log'

endpoint_fetch_products_n_details = LARAVEL_BASEURL + "/products/fetch_products_n_details"
endpoint_fetch_one_product_detail = LARAVEL_BASEURL + "/products/fetch_one_product_detail"
