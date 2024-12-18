import tiktoken

from constants.misc import SEPARATOR, TIkTOKEN_ENCODING

encoding = tiktoken.get_encoding(TIkTOKEN_ENCODING)
separator_len = len(encoding.encode(SEPARATOR))
