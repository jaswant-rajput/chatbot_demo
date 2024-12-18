import requests


def post(endpoint, data):
    return requests.post(endpoint, json=data)


def get(endpoint, params=None, headers=None, timeout=None):
    params_string = ""
    if params is not None:
        for key in params.keys():
            params_string += f"{key}={params[key]}&"
        params_string = params_string[:-1]

    endpoint_w_params = endpoint + "?" + params_string

    if timeout is not None:
        return requests.get(endpoint_w_params, timeout=10)
    else:
        return requests.get(endpoint_w_params)
