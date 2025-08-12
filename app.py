import streamlit as st
import os
import hashlib
import json
import fitz
import uuid
from datetime import datetime, timedelta
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index_name = "pdf-embeddings"

# Check if index exists, create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimension for text-embedding-3-small
        metric='cosine',
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Get the index
index = pc.Index(index_name)

# ---------------------- Memory Management Classes ----------------------
class ChatSession:
    def __init__(self, session_id: str, session_name: str = None):
        self.session_id = session_id
        self.session_name = session_name or f"Chat {session_id[:8]}"
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.message_count = 0
    
    def add_message(self, human_message: str, ai_message: str):
        self.memory.chat_memory.add_user_message(human_message)
        self.memory.chat_memory.add_ai_message(ai_message)
        self.last_activity = datetime.now()
        self.message_count += 1
    
    def get_chat_history(self):
        return self.memory.chat_memory.messages
    
    def get_formatted_history(self):
        messages = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return messages
    
    def clear_history(self):
        self.memory.clear()
        self.message_count = 0

class ChatSessionManager:
    def __init__(self):
        self.sessions = {}
        self.load_sessions()
    
    def create_session(self, session_name: str = None) -> str:
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id, session_name)
        self.sessions[session_id] = session
        self.save_sessions()
        return session_id
    
    def get_session(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            # Create new session if it doesn't exist
            self.sessions[session_id] = ChatSession(session_id)
            self.save_sessions()
        return self.sessions[session_id]
    
    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save_sessions()
    
    def get_all_sessions(self):
        # Sort sessions by last activity (most recent first)
        return sorted(
            self.sessions.values(),
            key=lambda x: x.last_activity,
            reverse=True
        )
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        sessions_to_remove = [
            sid for sid, session in self.sessions.items()
            if session.last_activity < cutoff_time
        ]
        for sid in sessions_to_remove:
            del self.sessions[sid]
        if sessions_to_remove:
            self.save_sessions()
        return len(sessions_to_remove)
    
    def save_sessions(self):
        """Save sessions to file (simplified version - in production use proper database)"""
        try:
            sessions_data = {}
            for sid, session in self.sessions.items():
                sessions_data[sid] = {
                    'session_name': session.session_name,
                    'created_at': session.created_at.isoformat(),
                    'last_activity': session.last_activity.isoformat(),
                    'message_count': session.message_count,
                    'messages': [
                        {
                            'type': type(msg).__name__,
                            'content': msg.content
                        }
                        for msg in session.get_chat_history()
                    ]
                }
            
            with open("chat_sessions.json", "w") as f:
                json.dump(sessions_data, f, indent=2)
        except Exception as e:
            st.error(f"Error saving sessions: {str(e)}")
    
    def load_sessions(self):
        """Load sessions from file"""
        try:
            with open("chat_sessions.json", "r") as f:
                sessions_data = json.load(f)
            
            for sid, data in sessions_data.items():
                session = ChatSession(sid, data['session_name'])
                session.created_at = datetime.fromisoformat(data['created_at'])
                session.last_activity = datetime.fromisoformat(data['last_activity'])
                session.message_count = data.get('message_count', 0)
                
                # Restore messages
                for msg_data in data.get('messages', []):
                    if msg_data['type'] == 'HumanMessage':
                        session.memory.chat_memory.add_user_message(msg_data['content'])
                    elif msg_data['type'] == 'AIMessage':
                        session.memory.chat_memory.add_ai_message(msg_data['content'])
                
                self.sessions[sid] = session
                
        except FileNotFoundError:
            # No existing sessions file
            pass
        except Exception as e:
            st.error(f"Error loading sessions: {str(e)}")

# Initialize session manager
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = ChatSessionManager()

# ---------------------- Core Functions ----------------------
def get_pdf_text(pdf_docs):
    text_data = []
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            text_data.append({
                "page_num": page_num,
                "content": page.get_text()
            })
    return text_data

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    
    # Use the correct LangChain Pinecone integration
    vectorstore = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key
    )
    return vectorstore

def get_qa_chain_with_memory():
    prompt_template = """
    You are an expert PDF assistant with deep knowledge of the provided documents. 
    Follow these guidelines to provide the best possible response:
    
    1. ALWAYS base your answer STRICTLY on the provided context and conversation history
    2. Use the conversation history to understand context and provide more personalized responses
    3. If the answer isn't in the context, say: "The answer is not available in the provided documents."
    4. Reference previous parts of the conversation when relevant
    5. For ambiguous questions, ask for clarification but suggest possible interpretations
    6. Structure complex answers with:
       - Key points first
       - Supporting details
       - Practical examples when applicable
    7. When relevant, include:
       - Key statistics
       - Important dates
       - Critical names/terms
    8. Maintain a professional yet approachable tone
    9. For comparative questions, present information in table format:
       | Aspect | Item A | Item B |
       |--------|--------|--------|
       | Feature 1 | ... | ... |
    
    Previous conversation:
    {chat_history}
    
    ---------------------
    Context:
    {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question", "chat_history"]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def generate_quiz_prompt(quiz_type, difficulty, num_questions, topic):
    """
    Generate appropriate quiz prompt based on difficulty level and quiz type
    """
    
    if difficulty.lower() == "easy":
        if quiz_type.lower() == "mcq" or quiz_type.lower() == "multiple choice":
            prompt_template = f"""
You are a professional quiz generator specializing in EASY-LEVEL educational assessments.

Your task is to create {num_questions} EASY-level multiple choice questions (MCQs) about "{topic}" using the context below.

🎯 **EASY LEVEL CHARACTERISTICS:**
- Focus on BASIC FACTS, DEFINITIONS, and SIMPLE RECALL
- Test fundamental concepts that are directly stated in the text
- Use straightforward vocabulary and simple sentence structures
- Questions should test "WHAT IS..." or "WHO/WHERE/WHEN" type information
- Avoid complex analysis, interpretation, or multi-step reasoning
- Target beginner-level understanding of the topic

📋 **QUESTION REQUIREMENTS:**
Each question must:
- Be crystal clear and unambiguous with simple language
- Test only ONE basic concept at a time
- Have 4 distinct answer options labeled A–D
- Include only ONE correct answer with 3 clearly wrong distractors
- Be phrased differently from others (absolutely no duplicates)
- Be directly based on facts explicitly mentioned in the provided context
- Avoid trick questions or overly complex wording
- Focus on direct information retrieval from the text

🚫 **WHAT TO AVOID:**
- Complex analytical questions
- Questions requiring interpretation or inference
- Multi-layered concepts
- Advanced terminology without explanation
- Questions combining multiple concepts

📝 **MANDATORY OUTPUT FORMAT:**
Q1. [Clear, simple question about basic concept]  
A) [Option A - clearly wrong]  
B) [Option B - clearly wrong]  
C) [Option C - correct answer]  
D) [Option D - clearly wrong]  
Answer: C

Q2. [Next question following same pattern]
... continue for all {num_questions} questions

📚 **Context about "{topic}":**  
{{context}}

**GENERATE EXACTLY {num_questions} EASY-LEVEL MCQs NOW.**
"""

        else:  # Short Answer
            prompt_template = f"""
You are a professional quiz generator specializing in EASY-LEVEL educational assessments.

Your task is to generate {num_questions} EASY-level short answer questions about "{topic}" based strictly on the provided context.

🎯 **EASY LEVEL CHARACTERISTICS:**
- Focus on BASIC FACTS, DEFINITIONS, and SIMPLE RECALL
- Test fundamental concepts directly stated in the text
- Ask for straightforward information that requires minimal interpretation
- Use simple, clear question formats like "What is...", "Define...", "List...", "Name..."
- Avoid complex reasoning or analytical thinking
- Target beginner-level understanding of the topic

📋 **QUESTION REQUIREMENTS:**
- Questions should be clear, direct, and unambiguous
- Test key basic facts, simple definitions, or fundamental concepts
- Avoid vague, broad, or overly complex questions
- Each question should focus on ONE simple concept
- Questions must be directly answerable from the provided context
- Use simple vocabulary appropriate for beginners

✍️ **ANSWER REQUIREMENTS:**
- Provide detailed, educational answers (4-5 complete sentences minimum)
- Include context and background information in answers
- Explain the concept clearly with examples from the text where possible
- Make answers comprehensive enough to help learning
- Use clear, simple language in explanations
- Ensure answers are factual and directly supported by the context

📝 **MANDATORY OUTPUT FORMAT:**
Q1. [Clear, simple question about basic concept]  
Answer: [Detailed 4-5 sentence explanation with context, examples, and clear information from the provided text. Include relevant background information to make the answer educational and comprehensive. Ensure the answer directly addresses the question while providing sufficient detail for learning purposes.]

Q2. [Next question following same pattern]
... continue for all {num_questions} questions

📚 **Context about "{topic}":**  
{{context}}

**GENERATE EXACTLY {num_questions} EASY-LEVEL SHORT ANSWER QUESTIONS NOW.**
"""

    elif difficulty.lower() == "medium":
        if quiz_type.lower() == "mcq" or quiz_type.lower() == "multiple choice":
            prompt_template = f"""
You are a professional quiz generator specializing in MEDIUM-LEVEL educational assessments.

Your task is to create {num_questions} MEDIUM-level multiple choice questions (MCQs) about "{topic}" using the context below.

🎯 **MEDIUM LEVEL CHARACTERISTICS:**
- Focus on APPLICATION OF CONCEPTS and CONNECTING IDEAS
- Test understanding through "HOW" and "WHY" questions
- Require students to make connections between different parts of the text
- Involve some interpretation and analysis of provided information
- Test conceptual understanding rather than just memorization
- Include questions that require applying knowledge to new situations

📋 **QUESTION REQUIREMENTS:**
Each question must:
- Test conceptual understanding and application of knowledge
- Require students to connect or compare different concepts from the text
- Have 4 distinct answer options labeled A–D with plausible distractors
- Include only ONE correct answer that demonstrates deeper understanding
- Be phrased differently from others (absolutely no duplicates)
- Go beyond simple recall to test comprehension and application
- May involve interpreting relationships between concepts
- Require moderate analytical thinking

🎯 **QUESTION TYPES TO INCLUDE:**
- "How does X relate to Y?"
- "Why is X important for Y?"
- "What would happen if..."
- "Which factor most influences..."
- "How can X be applied to..."
- "What is the relationship between..."

📝 **MANDATORY OUTPUT FORMAT:**
Q1. [Question testing conceptual understanding or application]  
A) [Plausible but incorrect option]  
B) [Plausible but incorrect option]  
C) [Correct answer showing deeper understanding]  
D) [Plausible but incorrect option]  
Answer: C

Q2. [Next question following same pattern]
... continue for all {num_questions} questions

📚 **Context about "{topic}":**  
{{context}}

**GENERATE EXACTLY {num_questions} MEDIUM-LEVEL MCQs NOW.**
"""

        else:  # Short Answer
            prompt_template = f"""
You are a professional quiz generator specializing in MEDIUM-LEVEL educational assessments.

Your task is to generate {num_questions} MEDIUM-level short answer questions about "{topic}" based strictly on the provided context.

🎯 **MEDIUM LEVEL CHARACTERISTICS:**
- Focus on APPLICATION, ANALYSIS, and CONNECTING CONCEPTS
- Test understanding through "HOW" and "WHY" questions
- Require interpretation and synthesis of information from the text
- Involve connecting different ideas or comparing concepts
- Test conceptual understanding and practical application
- May draw from multiple sections of the provided context

📋 **QUESTION REQUIREMENTS:**
- Questions should test conceptual understanding and application
- Require students to analyze, compare, or explain relationships
- May involve connecting ideas from different parts of the context
- Should test comprehension beyond simple recall
- Ask students to interpret or apply knowledge
- Focus on understanding processes, relationships, or implications

✍️ **ANSWER REQUIREMENTS:**
- Provide comprehensive, analytical answers (4-5 complete sentences minimum)
- Include explanations of relationships, processes, or applications
- Connect different concepts from the text where relevant
- Provide context for why the answer is important or significant
- Include examples or evidence from the provided context
- Demonstrate deeper understanding of the topic

🎯 **QUESTION TYPES TO INCLUDE:**
- "Explain how X influences Y..."
- "Why is X important for achieving Y?"
- "How do X and Y work together to..."
- "What are the implications of X for..."
- "Compare and contrast X and Y..."
- "Analyze the relationship between..."

📝 **MANDATORY OUTPUT FORMAT:**
Q1. [Question testing conceptual understanding, analysis, or application]  
Answer: [Comprehensive 4-5 sentence analytical answer that explains relationships, processes, or applications. Include relevant examples from the context and demonstrate deeper understanding of how concepts connect. Provide sufficient detail to show analytical thinking and conceptual comprehension.]

Q2. [Next question following same pattern]
... continue for all {num_questions} questions

📚 **Context about "{topic}":**  
{{context}}

**GENERATE EXACTLY {num_questions} MEDIUM-LEVEL SHORT ANSWER QUESTIONS NOW.**
"""

    elif difficulty.lower() == "hard":
        if quiz_type.lower() == "mcq" or quiz_type.lower() == "multiple choice":
            prompt_template = f"""
You are a professional quiz generator specializing in HARD-LEVEL educational assessments.

Your task is to create {num_questions} HARD-level multiple choice questions (MCQs) about "{topic}" using the context below.

🎯 **HARD LEVEL CHARACTERISTICS:**
- Focus on COMPLEX CONCEPTUAL UNDERSTANDING and CRITICAL THINKING
- Test advanced analytical, evaluative, and synthesis skills
- Require integration of multiple concepts from different sections
- Involve complex reasoning, evaluation, and critical analysis
- Test ability to apply knowledge to novel or complex situations
- May require understanding of implications, consequences, or theoretical frameworks

📋 **QUESTION REQUIREMENTS:**
Each question must:
- Test advanced conceptual understanding and critical thinking
- Require synthesis of information from multiple parts of the context
- Have 4 sophisticated answer options with subtle distinctions
- Include complex scenarios or theoretical applications
- Demand high-level analytical reasoning
- May involve evaluating arguments, comparing theories, or predicting outcomes
- Test ability to apply concepts to complex, real-world situations
- Require deep understanding of underlying principles

🎯 **ADVANCED QUESTION TYPES:**
- "Evaluate the effectiveness of X in achieving Y, considering..."
- "Analyze the complex relationship between X, Y, and Z..."
- "What would be the most significant consequence if X were modified to..."
- "Which theoretical framework best explains the interaction between..."
- "Critically assess the implications of X for..."
- "Synthesize the evidence to determine which approach would..."
- "Given the complexity of X, which factor would most critically influence..."

🧠 **CRITICAL THINKING ELEMENTS:**
- Multi-step logical reasoning
- Integration of concepts from different sections
- Evaluation of competing alternatives
- Analysis of cause-and-effect relationships
- Synthesis of complex information
- Application to novel scenarios

📝 **MANDATORY OUTPUT FORMAT:**
Q1. [Complex analytical question requiring critical thinking and synthesis]  
A) [Sophisticated but incorrect analysis]  
B) [Sophisticated but incorrect analysis]  
C) [Correct answer demonstrating deep understanding]  
D) [Sophisticated but incorrect analysis]  
Answer: C

Q2. [Next question following same pattern]
... continue for all {num_questions} questions

📚 **Context about "{topic}":**  
{{context}}

**GENERATE EXACTLY {num_questions} HARD-LEVEL MCQs NOW.**
"""

        else:  # Short Answer
            prompt_template = f"""
You are a professional quiz generator specializing in HARD-LEVEL educational assessments.

Your task is to generate {num_questions} HARD-level short answer questions about "{topic}" based strictly on the provided context.

🎯 **HARD LEVEL CHARACTERISTICS:**
- Focus on COMPLEX ANALYSIS, SYNTHESIS, and CRITICAL EVALUATION
- Test advanced conceptual understanding and theoretical application
- Require integration of multiple concepts from across the entire context
- Involve complex reasoning, critical thinking, and evaluative judgment
- Test ability to analyze implications, consequences, and theoretical frameworks
- May span multiple sections or chapters of the provided material

📋 **QUESTION REQUIREMENTS:**
- Questions must test advanced analytical and critical thinking skills
- Require synthesis and integration of complex information
- Should involve evaluation, critical analysis, or theoretical application
- May require drawing connections across multiple sections of the context
- Test deep understanding of underlying principles and frameworks
- Challenge students to think at the highest cognitive levels

✍️ **ADVANCED ANSWER REQUIREMENTS:**
- Provide sophisticated, analytical answers (4-5+ complete sentences minimum)
- Demonstrate complex reasoning and critical thinking
- Integrate multiple concepts and show their interrelationships
- Include analysis of implications, consequences, or theoretical applications
- Provide evidence-based arguments using information from the context
- Show deep understanding of complex principles and frameworks
- May include evaluation of competing perspectives or approaches

🎯 **COMPLEX QUESTION TYPES:**
- "Critically evaluate the effectiveness of X in addressing Y, considering the constraints..."
- "Analyze the complex interplay between X, Y, and Z and their collective impact on..."
- "Synthesize the evidence to construct an argument for why X represents the optimal approach to..."
- "Evaluate the theoretical implications of X for understanding Y, drawing from multiple perspectives..."
- "Assess the long-term consequences of implementing X, considering both benefits and potential risks..."
- "Compare and critically evaluate different approaches to X, analyzing their relative strengths and limitations..."

🧠 **CRITICAL THINKING REQUIREMENTS:**
- Multi-layered analysis involving several concepts
- Evaluation of competing theories or approaches
- Synthesis of information from different sections
- Assessment of implications and consequences
- Critical evaluation of effectiveness or validity
- Integration of theoretical and practical considerations

📝 **MANDATORY OUTPUT FORMAT:**
Q1. [Complex analytical question requiring synthesis, evaluation, or critical analysis across multiple concepts]  
Answer: [Sophisticated 4-5+ sentence analytical response that demonstrates critical thinking, integrates multiple concepts from the context, evaluates different perspectives or approaches, and provides evidence-based reasoning. Include analysis of implications, theoretical frameworks, or complex relationships between ideas. Show deep conceptual understanding and advanced analytical skills.]

Q2. [Next question following same pattern]
... continue for all {num_questions} questions

📚 **Context about "{topic}":**  
{{context}}

**GENERATE EXACTLY {num_questions} HARD-LEVEL SHORT ANSWER QUESTIONS NOW.**
"""

    else:
        # Default fallback (shouldn't reach here with proper validation)
        prompt_template = f"""
Invalid difficulty level specified. Please use: 'easy', 'medium', or 'hard'.
"""

    return prompt_template

# Complete function to replace your existing get_quiz_chain function
def get_quiz_chain(quiz_type, difficulty, num_questions, topic):
    """
    Generate quiz chain with difficulty-specific prompts
    """
    # Generate the appropriate prompt based on difficulty level
    prompt_template = generate_quiz_prompt(quiz_type, difficulty, num_questions, topic)
    
    # Create LangChain components
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, openai_api_key=openai_api_key)
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# ---------------------- Processing Functions with Memory ----------------------
def process_user_question(user_question, session_id):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    
    # Get the chat session
    session = st.session_state.session_manager.get_session(session_id)
    
    # Correct way to initialize existing Pinecone vector store
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    
    docs = vectorstore.similarity_search(user_question, k=3)
    chain = get_qa_chain_with_memory()
    
    # Get chat history in string format for the prompt
    chat_history = ""
    for msg in session.get_formatted_history()[-6:]:  # Last 6 messages for context
        role = "Human" if msg["role"] == "user" else "Assistant"
        chat_history += f"{role}: {msg['content']}\n"
    
    # Manually format the input for the chain
    context = "\n\n".join([doc.page_content for doc in docs])
    
    response = chain({
        "input_documents": docs, 
        "question": user_question,
        "chat_history": chat_history
    }, return_only_outputs=True)
    
    # Add to session memory
    session.add_message(user_question, response["output_text"])
    st.session_state.session_manager.save_sessions()
    
    return response["output_text"], docs

def generate_quiz(quiz_type, difficulty, num_questions, topic=None):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    
    # Correct way to initialize existing Pinecone vector store
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    
    # Use a default search term if no topic provided
    search_term = topic if topic else "main concepts key points"
    context_docs = vectorstore.similarity_search(search_term, k=5)
    
    chain = get_quiz_chain(quiz_type, difficulty, num_questions, topic)

    response = chain({"input_documents": context_docs}, return_only_outputs=True)
    
    # Split and limit questions
    questions = response["output_text"].split("\n\n")[:num_questions]
    return "\n\n".join(questions)

# Calculate unique hash for a file
def get_file_hash(file):
    hasher = hashlib.sha256()
    hasher.update(file.read())
    file.seek(0)  # Reset pointer after reading
    return hasher.hexdigest()

# Load previously embedded file hashes
def load_embedded_hashes():
    try:
        with open("embedded_files.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Save updated hash list
def save_embedded_hashes(hashes):
    with open("embedded_files.json", "w") as f:
        json.dump(hashes, f)

# ---------------------- Streamlit UI ----------------------
def main():
    st.set_page_config("📚 PDF Learning Assistant", layout="wide")
    st.header("📚 PDF Learning Assistant (Powered by OpenAI)")

    # Initialize session states
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "quiz_generated" not in st.session_state:
        st.session_state.quiz_generated = False
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = st.session_state.session_manager.create_session()

    with st.sidebar:
        st.title("📄 Document Setup")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type="pdf")

        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    embedded_hashes = load_embedded_hashes()
                    all_text = ""

                    for pdf in pdf_docs:
                        file_hash = get_file_hash(pdf)

                        if file_hash in embedded_hashes:
                            st.info(f"✅ Skipping {pdf.name} (already embedded)")
                            continue

                        # Extract and accumulate new content
                        pdf.seek(0)
                        reader = PdfReader(pdf)
                        for page in reader.pages:
                            all_text += page.extract_text()

                        # Mark as embedded
                        embedded_hashes.append(file_hash)

                    if all_text.strip() == "":
                        st.warning("No new files to embed.")
                    else:
                        # Chunk and embed only new content
                        chunks = get_text_chunks(all_text)
                        get_vector_store(chunks)
                        save_embedded_hashes(embedded_hashes)
                        st.success("✅ New PDFs embedded and saved!")

                    st.session_state.processed = True
            else:
                st.warning("Upload at least one PDF.")

        # Session Management Section
        st.title("💬 Chat Sessions")
        
        # New Chat Button
        if st.button("➕ New Chat", type="primary"):
            new_session_id = st.session_state.session_manager.create_session()
            st.session_state.current_session_id = new_session_id
            st.rerun()
        
        # Session List
        sessions = st.session_state.session_manager.get_all_sessions()
        
        if sessions:
            st.subheader("Your Chats")
            
            for session in sessions[:10]:  # Show last 10 sessions
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    is_current = session.session_id == st.session_state.current_session_id
                    button_label = f"{'🟢 ' if is_current else ''}{session.session_name}"
                    
                    if st.button(button_label, key=f"session_{session.session_id}"):
                        st.session_state.current_session_id = session.session_id
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{session.session_id}", help="Delete chat"):
                        st.session_state.session_manager.delete_session(session.session_id)
                        if session.session_id == st.session_state.current_session_id:
                            st.session_state.current_session_id = st.session_state.session_manager.create_session()
                        st.rerun()
                
                with col3:
                    st.caption(f"{session.message_count} msgs")
            
            # Cleanup old sessions
            if st.button("🧹 Cleanup Old", help="Remove sessions older than 24 hours"):
                removed_count = st.session_state.session_manager.cleanup_old_sessions()
                st.success(f"Removed {removed_count} old sessions")
                st.rerun()

    # Main content area
    tab1, tab2 = st.tabs(["💬 Chat with PDF", "❓ Generate Quiz"])

    with tab1:
        st.subheader("Ask a question about your document")
        
        # Current session info
        current_session = st.session_state.session_manager.get_session(st.session_state.current_session_id)
        st.caption(f"Current Chat: {current_session.session_name} | Messages: {current_session.message_count}")
        
        # Display chat history
        if current_session.message_count > 0:
            st.subheader("Chat History")
            messages = current_session.get_formatted_history()
            
            # Create a container for chat messages
            chat_container = st.container()
            with chat_container:
                for i, message in enumerate(messages):
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.write(message["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.write(message["content"])
        
        # Clear chat history button
        if current_session.message_count > 0:
            if st.button("🗑️ Clear Chat History"):
                current_session.clear_history()
                st.session_state.session_manager.save_sessions()
                st.rerun()
        
        # Question input with form to prevent auto-rerun
        with st.form(key="question_form", clear_on_submit=True):
            user_question = st.text_input("Your question:", key="chat_input")
            submit_button = st.form_submit_button("Send")

        if submit_button and user_question and st.session_state.processed:
            with st.spinner("Searching answer..."):
                try:
                    answer, sources = process_user_question(user_question, st.session_state.current_session_id)
                    
                    # Display the new answer immediately
                    st.success("✅ Question processed and added to chat history!")
                    
                    with st.expander("View Sources"):
                        for i, doc in enumerate(sources):
                            st.caption(f"Source {i+1}: {doc.page_content[:300]}...")
                    
                    # Force refresh of chat history display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
        
        elif submit_button and user_question and not st.session_state.processed:
            st.warning("Please process PDFs first to enable chat functionality.")
        
        elif submit_button and not user_question:
            st.warning("Please enter a question.")

    with tab2:
        if st.session_state.processed:
            st.subheader("Generate a Quiz")
            col1, col2, col3 = st.columns(3)
            with col1:
                quiz_type = st.selectbox("Question Type", ["MCQ", "Short Answer"])
            with col2:
                difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
            with col3:
                num_q = st.slider("Number of Questions", 1, 10, 5)

            topic = st.text_input("Specific Topic (optional)")

            if st.button("Generate Quiz"):
                with st.spinner("Generating quiz..."):
                    try:
                        quiz = generate_quiz(quiz_type, difficulty, num_q, topic)
                        st.session_state.quiz_generated = True
                        st.session_state.quiz_content = quiz
                    except Exception as e:
                        st.error(f"Error generating quiz: {str(e)}")

            if st.session_state.quiz_generated:
                st.subheader("Generated Quiz")
                st.code(st.session_state.quiz_content, language="markdown")
                st.download_button(
                    "Download Quiz",
                    st.session_state.quiz_content,
                    file_name=f"{difficulty}_{quiz_type}_quiz.txt",
                    mime="text/plain"
                )
        else:
            st.info("Please process PDFs to enable quiz generation.")

if __name__ == "__main__":
    main()