from pathlib import Path

DEBUG = True

cwd = str(Path(__file__).parent.parent.absolute())

ARTICLE_MAX_TOKENS = 600
CHATGPT_MAX_TOKENS = 1500
MAX_TOKEN_BUFFER = 50

VECTOR_DATAS_DIR = "data"
STEP_SIZE = 1000

MAX_SECTION_LEN = 1000
SEPARATOR = "\n* "
TIkTOKEN_ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002


version = 2

dataset_csv_file = "dataset_qa_final_processed_w_tokens.csv"

ERROR_LOG_PATH="logs/error_{time:DD-MM-YYYY}.log"
INFO_LOG_PATH="logs/info_{time:DD-MM-YYYY}.log"