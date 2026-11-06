import os 
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START , END 

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)
def merge_score_dicts(existing :dict , newupdate : dict) -> dict:
    if existing is None:
        return newupdate 
    return {**existing , **newupdate}

#create a state 
class AnalyzerState(TypedDict):
    raw_text : str 
    safety_scores : Annotated[dict[str , int],merge_score_dicts]


#nodes 
def toxicity_node(state: AnalyzerState) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under our single state key
    return {"safety_scores": {"toxicity_level": score}}

def copyright_node(state: AnalyzerState) -> dict:
    print("\n🔏 [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"copyright_risk": score}}


def culture_node(state: AnalyzerState) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"cultural_insensitivity": score}}


builder = StateGraph(AnalyzerState)


builder.add_node("toxicity_node",toxicity_node)
builder.add_node("copyright_check",copyright_node)
builder.add_node("culture_node",culture_node)


builder.add_edge(START,"toxicity_node")
builder.add_edge(START,"copyright_check")
builder.add_edge(START,"culture_node")


builder.add_edge("toxicity_node",END)
builder.add_edge("copyright_check",END)
builder.add_edge("culture_node",END)


app = builder.compile()

sample_script = """
    Hey cricket lovers, kya aap bhi soch rahe ho ki Virat Kohli ka birthday itna epic kyun hota hai? 🎉 Toh aaj hum aapko le chalenge ek aise world tour pe jahan family ki secret traditions se leke woh hidden facts tak sab kuch reveal hoga—jo sirf uske close circle ko hi pata hain!  

Sabse pehle, picture karo: Virat ki mom, Anju, jab birthday cake cut karti hain, woh ek simple “Happy Birthday, beta, tumhara hard work hi tumhe aaj ke star banata hai” ke saath ek special wish deti hain—aur usmein ek choti si mantra hoti hai, “Stay humble, stay hungry.” Ye line har baar uske liye ek power‑boost ki tarah kaam karti hai.  

Ab baat karte hain uss secret family tradition ki—Har saal birthday ke pehle raat, poora parivaar milke ek choti si “gratitude circle” banata hai. Sab log ek dusre ko ek line mein thank you bolte hain, aur uske baad Virat ko ek handwritten note milta hai jisme uske bachpan ki koi funny memory hoti hai. Yeh traditionusko grounded rakhta hai, chahe wo stadium ke sabse bade spotlight mein ho.  

Aur sabse shocking part—team mates ke liye ek surprise! 🎁 Virat ne ek private video montage arrange kiya tha, jisme har teammate apni favorite cricket momentke saath ek short message record karta hai. Lekin is baar, ek extra twist tha: us video ke end mein ek hidden camera se uski childhood cricket ground ki live feed aayi, jahan uske bachpan ke friends ne ek flash mob kiya—complete with “Kohli ke liye” banner! Yeh sab uske teammates ko bilkul surprise kar gaya, aur unke liye ek unforgettable memory ban gaya.  

Toh bas, agle 2 minutes mein aapko milega pura insider scoop—family ki love, mom ki heartfelt wish, aur woh surprise jo har kisi ko hairaan kar dega. Miss matkaro, kyunki yeh content sirf yahi milta hai! 🚀  

Ready ho? Let’s dive right in and celebrate Virat’s birthday the ultimate way—Hinglish style, high energy, full on fun! 🎊
    """
    

    
initial_state = {
    "raw_text": sample_script,
    "safety_scores": {} # Initialized as an empty dictionary
}
    
final_state = app.invoke(initial_state)
    

print(final_state["safety_scores"])
