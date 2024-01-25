#!pip install milvus pymilvus nltk llama_index openai python-dotenv requests docx2txt gradio
import gradio as gr
import nltk
import ssl
import dotenv
import os

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = ""
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def query_aksarc_documentrepository(input_query):
    nltk.download('punkt')
    nltk.download('stopwords')
    
    from llama_index import(
        VectorStoreIndex,
        SimpleKeywordTableIndex,
        SimpleDirectoryReader,
        LLMPredictor,
        ServiceContext,
        StorageContext
    )
    #from langchain.llms.openai import OpenAIChat
    from langchain.chat_models import ChatOpenAI
    from llama_index.vector_stores import MilvusVectorStore 
    from milvus import default_server
    default_server.start()
    vector_store = MilvusVectorStore(
        host = "127.0.0.1",
        port = default_server.listen_port,
        dim=1536
    )
    topics = ["install", "update", "other"]
    # Load all aks documents
    aks_docs = {}
    for topic in topics:
        aks_docs[topic] = SimpleDirectoryReader(input_dir=f"./data/{topic}/").load_data()
    
    #from llama_index.llms import OpenAI
    #llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    #service_context = ServiceContext.from_defaults(llm=llm)
    llm_predictor_chatgpt = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Build topic document index
    topic_indices = {}
    index_summaries = {}
    for topic in topics:
        topic_indices[topic] = VectorStoreIndex.from_documents(aks_docs[topic], service_context=service_context, storage_context=storage_context)
        # set summary text for topic
        index_summaries[topic] = f"AKS Arc documents about {topic}"
    from llama_index.indices.composability import ComposableGraph
    graph = ComposableGraph.from_indices(
        SimpleKeywordTableIndex,
        [index for _, index in topic_indices.items()], 
        [summary for _, summary in index_summaries.items()],
        max_keywords_per_chunk=50
    )
    from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
    decompose_transform = DecomposeQueryTransform(
        llm_predictor_chatgpt, verbose=True
    )

    from llama_index.query_engine.transform_query_engine import TransformQueryEngine
    custom_query_engines = {}
    for index in topic_indices.values():
        query_engine = index.as_query_engine(service_context=service_context)
        transform_extra_info = {'index_summary': index.index_struct.summary}
        tranformed_query_engine = TransformQueryEngine(query_engine, decompose_transform, transform_metadata=transform_extra_info)
        custom_query_engines[index.index_id] = tranformed_query_engine

    custom_query_engines[graph.root_index.index_id] = graph.root_index.as_query_engine(
        retriever_mode='simple', 
        response_mode='tree_summarize', 
        service_context=service_context
    )

    query_engine_decompose = graph.as_query_engine(custom_query_engines=custom_query_engines,)
    response_chatgpt = query_engine_decompose.query(input_query)
    return response_chatgpt

user_query = gr.components.Textbox(lines=2, label="Input Query")
output_summary = gr.components.Textbox(label="Summary")

Interface = gr.Interface(
    fn=query_aksarc_documentrepository,
    inputs=user_query,
    outputs=output_summary,
    title="AKS Arc Document Query Engine",
    description="Query AKS Arc document repository",
).launch(share=True)