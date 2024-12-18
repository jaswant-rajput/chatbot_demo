

SMALL_TALK = 0
PRODUCT_RELATED_QUERY = 1
NON_PRODUCT_RELATED_QUERY = 2
SHARING_FEEDBACK = 3
ANSWERING_QUESTION = 4
END_CONVERSATION = 5

intent_map = {
    SMALL_TALK: "SMALL_TALK",
    PRODUCT_RELATED_QUERY: "PRODUCT_RELATED_QUERY",
    NON_PRODUCT_RELATED_QUERY: "NON_PRODUCT_RELATED_QUERY",
    SHARING_FEEDBACK: "SHARING_FEEDBACK",
    ANSWERING_QUESTION: "ANSWERING_QUESTION",
    END_CONVERSATION: "END_CONVERSATION",
}

INTENT_ESTIMATOR_PROMPT = f"""Given the conversation below, where last message is latest message from user.
Respond with {SMALL_TALK} if user is making small talk unrelated to the organization and its services,
{PRODUCT_RELATED_QUERY} if user is asking a question about a product,
{NON_PRODUCT_RELATED_QUERY} if user is asking a non-product related question,
{SHARING_FEEDBACK} if user is sharing feedback,
{ANSWERING_QUESTION} if user is answering a question,
{END_CONVERSATION} if user is ending the conversation.
"""

ANSWER_WITH_CONTEXT_STRICT_PROMPT= "Respond with the following JSON format:\n {\n\"response\": \"<Insert your response here>\",\n\"CONTEXT adherence\": \"<Explain whether the response was generated using info mentioned in CONTEXT or not. Explain why you think so.>\",\n\"prompt adherence\": \"<Explain whether your response adheres to the instructions in the system prompt. Explain why you think so.>\",\n\"is_answered\": \"<True/False>// Was the response able to answer user's query satisfactorily or not. No explanation needed.>\"\n}" # type: ignore



CITATION_QA_TEMPLATE = (
    "Please provide an answer based solely on the provided sources."
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every sentence should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it and in the square bracket format. "
    "If none of the sources are helpful, return {unsure_msg}."
    "For example:\n"
    "154:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "657:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [657], "
    "which occurs in the evening [154].\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
)


CITATION_REFINE_TEMPLATE = (
    "You are provided with an existing answer and its sources."
    "Please refine the answer, ensuring that the citations are correct and appropriately placed. "
    "Use the provided sources and cite them using their corresponding numbers in square brackets. "
    "Every sentence should include at least one source citation. "
    "If the original answer does not reference a source when it should, or cites the wrong source, correct it. "
    "\nFor Example:\n"
    "\n---Example Sources Start---\n"
    "154:\n"
    "The sky is red in the evening and blue in the morning.\n\n"
    "657:\n"
    "Water is wet when the sky is red.\n\n"
    "892:\n"
    "The grass is green when it rains.\n\n"
    "213:\n"
    "Cats are known to be independent creatures.\n\n"
    "874:\n"
    "Dogs are often seen as loyal companions.\n\n"
    "462:\n"
    "Cats and dogs can live harmoniously together with proper training.\n\n"
    "---Example Sources End---\n"
    "---Example Query Starts---\n"
    "Existing Answer: Water will be wet when the sky is red, which occurs in the evening [657: Water is wet when the sky is red].\n"
    "Refined Answer: Water will be wet when the sky is red [657], which occurs in the evening [154].\n"
    "Existing Answer: Cats and dogs can live together with proper training [462; harmoniously].\n"
    "Refined Answer: Cats and dogs can live together with proper training [462] and are known to form unique bonds [213] [874].\n"
    "---Example Query Ends---\n"
    "Now its your turn.\n"
    "Below are several numbered sources of information. "
    "\n---Sources Start---\n"
    "{context_msg}"
    "\n---Sources End---\n"
    # "\n---Conversation Start---\n"
    # "{conversation}"
    # "\n---Conversation End---\n"
    "Existing Answer: {existing_answer}\n"
    "Refined Answer: "
)
