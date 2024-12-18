from pathlib import Path

cwd = str(Path(__file__).parent.absolute())


BACKGROUND_FLASK_ENDPOINT = "https://api.example.com"
LARAVEL_BASEURL = "https://api.example.com/api"

# redis queue related
REDIS_HOST = "localhost"
REDIS_PORT = 6379

DEBUG = True # Should be true if testing locally

VECTOR_DATAS_DIR = "vector_datas"

