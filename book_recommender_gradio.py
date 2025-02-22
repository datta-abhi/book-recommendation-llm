import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

#langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

#loading the books data
books = pd.read_csv('base_data_books.csv')

# thumbnail for creation
books['mod_thumbnail'] = books['thumbnail'] + "&fife=w400"
books['mod_thumbnail'] = np.where(books['mod_thumbnail'].isna(),'default_cover.png', books['mod_thumbnail'])

# loading our vectorstore
DB_NAME = 'books-vector-db'

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing Chroma vectorstore
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

def retrieve_semantic_recommendations(query: str,
                                      category:str = None,
                                      tone: str = None,
                                      initial_top_k:int = 20,
                                      final_top_k:int = 6)-> pd.DataFrame:
    
    """
    searches for books with similar descriptions using vector similarity,
    filters by category if specified
    sorts by emotional tone,
    returns a dataframe of recommended books
    """
    # semantic search of query on our vectorstore
    recs = vectorstore.similarity_search(query,k=initial_top_k)
    # book isbns from our books df
    isbns_list = [int(i.page_content.strip('"').split(':')[0]) for i in recs]
    book_recs = books[books['isbn13'].isin(isbns_list)].head(initial_top_k)
    
    if category!= 'All':
        book_recs = book_recs[book_recs['new_category']==category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)
    
    # tone search
    tone_list = ["anger", "fear", "joy", "sadness", "surprise", "neutral"]
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
        
    return book_recs       

def recommend_books(query:str,category:str,tone:str):
    """
    retrieves recommendations, formats book description within 30 words,
    returns a list of tuples (thumbnail_image,formatted caption)
    """
    recs = retrieve_semantic_recommendations(query,category,tone)
    results = []
    
    #edge case if the there is no books in a category after similarity search
    if recs.empty:
        return [('default_cover.png',"No recommendations found in given category for your description")]    
        
    for _,row in recs.iterrows():
        description = row['description']
        # first 30 words of description for long descriptions
        short_descr = " ".join(description.split()[:30])+ '...'
        
        # give author names upto 2 authors
        authors_split = row['authors'].split(";")
        if len(authors_split)==2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split)>2:
            authors_str = f"{authors_split[0]}, {authors_split[1]} and others"
        else:
            authors_str = row['authors']
        
        # caption combining description and author names    
        caption = f"{row['title']} by {authors_str}: {short_descr}"
        results.append((row['mod_thumbnail'],caption))
        
    return results                 
    
# creating the Gradio UI
categories = ["All"] + list(books['new_category'].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Ocean()) as dashboard:
    gr.Markdown("# MyBuddy Book Recommender")
    
    with gr.Row():
        user_query = gr.Textbox(label="Please what you want to read about",
                                placeholder = "eg: A story about friendship")
        category_dr = gr.Dropdown(choices= categories, label = "Select category",value = 'All')
        tone_dr = gr.Dropdown(choices= tones, label = "Select emotional tone",value = 'All')
        submit_btn = gr.Button("Recommend me books")
        
    gr.Markdown("## You'll love these books")
    output = gr.Gallery(label="Recommended books",columns = 3,rows = 2, 
                        object_fit="contain")  #we show 6 books
    
    submit_btn.click(fn = recommend_books,
                     inputs = [user_query,category_dr,tone_dr],
                     outputs=output)    
               

if __name__ == "__main__":
    dashboard.launch()    