import os, json, glob
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeStore
import pinecone, dotenv

dotenv.load_dotenv()

file = glob.glob("data_raw/data.pdf")

loader = PyPDFLoader if file.endswith(".pdf") else UnstructuredFileLoader
docs = sum((loader(fp).load() for fp in glob.glob("data_raw/*")), [])

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = splitter.split_documents(docs)

emb = OpenAIEmbeddings(model="text-embedding-ada-002")
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter")
index = pinecone.Index("mh-agent", dimension=1536)      # or create if not exists
store = PineconeStore.from_documents(chunks, emb, index_name="mh-agent")  

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.moderation import OpenAIModerationChain   # safety
from langchain import create_retrieval_chain                    # new LCEL helper
from langchain.prompts import ChatPromptTemplate

moderation = OpenAIModerationChain()                            # passes/blocks input :contentReference[oaicite:2]{index=2}
retriever = store.as_retriever(search_kwargs=dict(k=4))

SYSTEM_PROMPT = """You are a supportive mental-health assistant. 
Use the retrieved information to answer with empathy.
If user asks for medical diagnosis or self-harm instructions, 
politely refuse and suggest professional help."""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT),
     ("context", "{context}"),
     ("user", "{input}")]
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

rag_chain = create_retrieval_chain(llm, retriever, prompt=prompt)   :contentReference[oaicite:3]{index=3}
memory = ConversationBufferMemory(return_messages=True)

def agent(user_input: str):
    moderation.check(user_input)        # raises if flagged
    result = rag_chain.invoke({"input": user_input}, memory=memory)
    return result["answer"]

