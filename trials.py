from operator import itemgetter
import string
import random

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


history = get_by_session_id("1")
history.add_message(AIMessage(content="hello"))
history.add_message(AIMessage(content="what is going on boy "))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at answring simply "),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

llm=ChatGroq(temperature=0,
            model_name="llama3-8b-8192",
            api_key=os.getenv("GROQ_API_KEY"))

chain=prompt|llm|StrOutputParser()

def get_conversational():
    chain_with_history = RunnableWithMessageHistory(
        chain,
        # Uses the get_by_session_id function defined in the example
        # above.
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_history

def getSessionId():
    """
    generates a random session id using random caracters and random numbers

    Returns:
        sessionId(str): combination of 12 random charactes and numbres  
    """
    chars= string.ascii_letters + string.digits
    sessionId= ''.join(random.choice(chars) for i in range(12))
    return sessionId 

def getConfig():
    """
    genreates a random configuration for the session

    Retruns:
        config(dict): configuration used by the conversationalRagChain
    """
    sessionId=getSessionId()
    config={"configurable":{"session_id":sessionId}}
    return config


conversation=get_conversational()
print(conversation.invoke(
    {"question": "this is a question"},
    config=getConfig()
))

print(store)


# print("----------------------------------------")
# for chunk in conversation.stream(
#     {"question": "What are the main goals of the Webb telescope?"},
#     config={"configurable": {"session_id": "user_abc_123"}}
# ):
#     # chunk is a string containing a small piece of the full answer
#     print(chunk, end="", flush=True)

# print("----------------------------------------")
# print()
# print(store)