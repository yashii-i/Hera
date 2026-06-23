import gradio as gr
import random
from datetime import date

from huggingface_hub import InferenceClient
#!pip install -q sentence-transformers
from sentence_transformers import SentenceTransformer
import torch

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

with open("knowledge.txt", "r", encoding="utf-8") as file:
  knowledge_base = file.read()

def preprocess_text(text):
  cleaned_text = text.strip()
  chunks = cleaned_text.split("\n")
  cleaned_chunks = []
  for chunk in chunks:
    stripped_chunk = chunk.strip()
    if len(stripped_chunk) > 0:
      cleaned_chunks.append(stripped_chunk)
  return cleaned_chunks
cleaned_chunks = preprocess_text(knowledge_base)


model = SentenceTransformer('all-MiniLM-L6-v2')
def create_embeddings(text_chunks):
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) 
  return chunk_embeddings
chunk_embeddings = create_embeddings(cleaned_chunks) 

#making a function to find similarities bw query and chunks 
def get_top_chunks(query, chunk_embeddings, text_chunks):
  query_embedding = model.encode(query, convert_to_tensor=True) 
  query_embedding_normalized = query_embedding / query_embedding.norm()
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)
  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized)
  top_indices = torch.topk(similarities, k=3).indices
  top_chunks = []
  for i in top_indices:
    chunk = text_chunks[i]
    top_chunks.append(chunk)
  return top_chunks


def respond(message, history):

    top_chunks = get_top_chunks(message, chunk_embeddings, cleaned_chunks)
    context = "\n".join(top_chunks)

    messages = [{"role": "system", "content": """You are a STEAM Opportunity Advisor (Hera) for girls and women. You are Hera, an AI career and opportunity advisor for girls and women in STEAM.
Help users find scholarships, internships, competitions, courses, and clubs based ONLY on their stated interests.
Keep responses under 120 words.
One-shot Example
User: I like science but I don’t know what to do.
Hera:
 That’s a great starting point in STEAM. What part of science interests you most — space, biology, chemistry, or tech?
Once I know, I can suggest beginner-friendly courses, competitions, or programs you can join."""}]


    if history:
        messages.extend(history)
    

    messages.append({
    "role": "user",
    "content": f"Context:\n{context}\n\nQuestion:\n{message}"
    })
    
    ##return response.choices[0].message.content.strip()
    
    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=150,
        temperature=1,
        top_p=0.5, 
        stream = True
    ): 
    
        token = message.choices[0].delta.content
        response += token
        yield response
    
quotes = [
    "Something is better than nothing",
    "Consistency, consistency, consistency",
    "Know your content and know it well",
    "Hardwork does not speak for itself, you do",
    "The biggest adventure you can take is to live the life of your dreams.",
    "It's not half as impossible as everyone assumes.",
    "Good things fall apart so better things can come together"
]
random.seed(str(date.today()))
daily_quote = random.choice(quotes)

#tracker
tracked_opportunities = []

def add_opportunity(name, status, details):
    if not name.strip():
        return tracked_opportunities, name, status, details
    tracked_opportunities.append([name, status, details])
    return tracked_opportunities, "", "Interested", ""

custom_css = """
.gradio-container { background-color: #f0f4ff !important; }
input, textarea { background-color: #eff6ff !important; border-color: #93c5fd !important; color: #1e1b4b !important; }
button.primary { background-color: #f79d65 !important; color: #c8b6ff !important; }
.block { border-color: #c4b5fd !important; background-color: #fff !important; }
label, .label-wrap, em, .md, .prose { color: #3b0764 !important; }
h1, h2, h3, .block-title { color: #b8c0ff!important; }
.message.bot, .message.bot p, .message.bot span, .bot { color: #9333ea !important; }
.message.user, .message.user p, .message.user span { color: #1e1b4b !important; }
label, .block label, .label-wrap span { color: #a2d2ff !important; }
"""

with gr.Blocks(theme="hmb/amethyst", css=custom_css) as demo:
    gr.Image(value="hera banner.png", show_label=False, elem_id="top-image")
    gr.Markdown(f"""
    ## 🌟 Daily Motivation
    *"{daily_quote}"*
    """)
    gr.ChatInterface(respond)
    
    gr.Markdown("---")
    gr.Markdown("## 📋 Tracker")

    with gr.Row():
        with gr.Column():
            opp_name = gr.Textbox(label="Opportunity Name", placeholder="e.g., NASA Internship, Google Scholarship")
            opp_status = gr.Dropdown(choices=["Interested", "Applied","Deciding","Not Interested",""], value="", label="Your Status")
        with gr.Column():
            opp_details = gr.TextArea(label="Important Details", placeholder="Deadlines, requirements, links...")
            
    submit_btn = gr.Button("Add to Tracker List", variant="primary")
    
    gr.Markdown("### Your Opportunities")
    tracker_table = gr.Dataframe(
        headers=["Opportunity", "Status", "Important Details"],
        datatype=["str", "str", "str"],
        wrap=True
    )
    
    submit_btn.click(
        fn=add_opportunity,
        inputs=[opp_name, opp_status, opp_details],
        outputs=[tracker_table, opp_name, opp_status, opp_details]
    )

    demo.launch()

##chatbot.launch(debug=True)