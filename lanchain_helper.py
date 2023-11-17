from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.embeddings import GPT4AllEmbeddings

#If using hugging face API instead of running LLM in local
#import os
#from langchain import HuggingFaceHub
#from langchain.embeddings import HuggingFaceEmbeddings
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = "ADD KEY HERE"
#embeddings = HuggingFaceEmbeddings()


embeddings = GPT4AllEmbeddings() 

def embbed_txt(file):
    with open(file, 'r') as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 650,
        chunk_overlap  = 20,
        length_function = len,
        add_start_index = True,
    )    
    splitted = text_splitter.create_documents([text])
    db = FAISS.from_documents(splitted, embeddings)
    return db


def query(question, k=3):
    db = embbed_txt("./data/cv.txt")
    docs = db.similarity_search(question, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    print("Loading LLM")
    llm = GPT4All(model ="./models/mistral-7b-openorca.Q4_0.gguf", verbose=False)
    #llm = CTransformers(model="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral")
    #llm = HuggingFaceHub(repo_id="google/flan-ul2", model_kwargs={"temperature":0.1, "max_length": 64})


    print("LLM loaded") 
    #prompt template 
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template=
        """
        You are a helpful assistant on my CV website. Your role is to answer the question asked by potential recruiters about me, Julien GODFROY
        
        Answer the following question: {question}
        By searching in my CV here: {docs}
        
        Only use the factual information from the transcript to answer the question. Be positive ! Your goal is to convince the recruiter to hire me.
        
        If you feel like you don't have enough information to answer the question, you can imaginate an answer linked to my CV.
        
        Your answers should be detailed, developped and postive.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    print("Starting chain")
    response = chain.run(question=question, docs=docs_page_content)
    print("Chain finished")
    response = response.replace("\n", "")
    return response


if __name__ == "__main__":
    
    print(query("Why should I hire Julien ?" )[0])