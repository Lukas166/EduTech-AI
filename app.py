import streamlit as st
import json
import re
import os
import urllib.parse
from bots import RAGChatbot # Assuming bot.py is in the same directory

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Platform Edukasi AI", page_icon="🎓")

# --- Global Variables / Data Loading ---
COURSES_DATA = [] # Populated by load_and_parse_courses_from_json
COURSE_CONTENT_DETAILS = {} # Populated by load_and_parse_courses_from_json
COURSES_PER_PAGE_DASHBOARD = 3

# --- Helper Functions for Data Loading and Parsing ---
def parse_course_content(content_text):
    """Parses the raw content string into a structured dictionary."""
    parsed = {
        "overview": [],
        "key_concepts": [],
        "code_example": {"language": "python", "code": ""},
        "applications": []
    }
    
    # Split content into logical blocks (paragraphs or sections)
    # Using a more robust split that handles multiple newlines and retains some structure
    blocks = re.split(r'\n\s*\n+', content_text.strip())
    
    current_section = "overview" # Default section

    for block in blocks:
        block_lower = block.lower()
        
        if block_lower.startswith("key concepts:"):
            current_section = "key_concepts"
            # Remove the header from the block content
            concept_list_str = block[len("key concepts:"):].strip()
            if concept_list_str:
                # Assuming concepts are listed with '-' or '* ' or just newlines
                parsed["key_concepts"] = [c.strip() for c in re.split(r'\n\s*[-*]?\s*', concept_list_str) if c.strip()]
        elif block_lower.startswith("code example:"):
            current_section = "code_example"
            # Extract language and code block
            match_lang_code = re.search(r"code example:\s*(\w+):\s*\n```\w*\n([\s\S]*?)\n```", block, re.IGNORECASE | re.DOTALL)
            if match_lang_code:
                parsed["code_example"]["language"] = match_lang_code.group(1).strip().lower()
                parsed["code_example"]["code"] = match_lang_code.group(2).strip()
            else: # Fallback if regex fails, add as overview
                 parsed["overview"].append(block)
        elif block_lower.startswith("applications:"):
            current_section = "applications"
            app_list_str = block[len("applications:"):].strip()
            if app_list_str:
                 parsed["applications"] = [a.strip() for a in re.split(r'\n\s*[-*]?\s*', app_list_str) if a.strip()]
        else:
            # If it's not a special section header, add to current or default to overview
            if current_section == "overview":
                 parsed["overview"].append(block.strip())
            # If it's part of a section already being processed (e.g. more concepts/apps without new header)
            # This part might need more sophisticated logic if sections are interleaved without headers.
            # For now, unheadered blocks go to overview unless a section was just started.

    # Join overview paragraphs
    parsed["overview"] = "\n\n".join(parsed["overview"])
    
    return parsed

def load_and_parse_courses_from_json(file_path="dataWeb.json"):
    """Loads courses from JSON and populates global data structures."""
    global COURSES_DATA, COURSE_CONTENT_DETAILS
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_courses = data.get("courses", [])
        
        temp_courses_data = []
        temp_course_content_details = {}
        
        for course_json in raw_courses:
            course_id = course_json.get("id")
            raw_content = course_json.get("content", "")
            
            if not course_id:
                continue

            # Create entry for COURSES_DATA (for cards)
            title = course_id.replace("_", " ").replace("-", " ").title()
            
            # Extract description (e.g., first few sentences or up to a certain length)
            # Simple extraction: first two non-empty lines.
            content_lines = [line.strip() for line in raw_content.split('\n') if line.strip()]
            description = " ".join(content_lines[:2]) if len(content_lines) > 1 else (content_lines[0] if content_lines else "No description available.")
            description = (description[:200] + '...') if len(description) > 200 else description


            # Extract topics from "Key Concepts" if possible, else use generic tags or id
            parsed_content_for_topics = parse_course_content(raw_content) # Parse once
            topics = parsed_content_for_topics.get("key_concepts", [])[:4] # Take first 4 key concepts as topics
            if not topics: # Fallback topics
                topics = [tag.strip() for tag in course_id.split('_')][:4]


            temp_courses_data.append({
                "id": course_id,
                "title": title,
                "description": description,
                "topics": topics 
            })
            
            # Store parsed content for detail view
            temp_course_content_details[course_id] = parsed_content_for_topics
            
        COURSES_DATA = temp_courses_data
        COURSE_CONTENT_DETAILS = temp_course_content_details
        
        if not COURSES_DATA:
            st.error(f"No courses loaded from {file_path}. Please check the file format and content.")

    except FileNotFoundError:
        st.error(f"Error: The file {file_path} was not found.")
        COURSES_DATA = []
        COURSE_CONTENT_DETAILS = {}
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {file_path}. Please check for syntax errors.")
        COURSES_DATA = []
        COURSE_CONTENT_DETAILS = {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading courses: {e}")
        COURSES_DATA = []
        COURSE_CONTENT_DETAILS = {}

# --- Chatbot Initialization ---
@st.cache_resource # Cache the chatbot instance
def get_chatbot_instance():
    # Ensure bot.py, faq.json, and best_embedding_model are accessible
    # GEMINI_API_KEY should be in .env file or environment variables
    try:
        # Construct absolute paths if necessary, assuming files are in the same dir as app.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        faq_file_path = os.path.join(script_dir, "faq.json")
        model_path_dir = os.path.join(script_dir, "best_embedding_model")

        if not os.path.exists(faq_file_path):
            st.error(f"FAQ file not found: {faq_file_path}. Chatbot might not function correctly.")
            # You might want to return a dummy chatbot or raise an error
        if not os.path.isdir(model_path_dir):
            st.error(f"Model directory not found: {model_path_dir}. Chatbot might not function correctly.")

        chatbot = RAGChatbot(faq_file=faq_file_path, model_path=model_path_dir)
        print("RAGChatbot instance created.")
        return chatbot
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}. Please ensure all dependencies and API keys are set.")
        print(f"Chatbot initialization error: {e}")
        return None

# --- CSS Kustom ---
# (CSS remains largely the same as your last version, with minor adjustments if needed)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(-45deg, #0f0f0f, #1a1a1a, #2d1b69, #1e3a8a, #0f172a, #1e1b4b);
        background-size: 400% 400%;
        animation: darkGradientBG 25s ease infinite;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    @keyframes darkGradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main .block-container {
        background: rgba(15, 23, 42, 0.3);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
        margin-top: 1rem;
    }

    .css-1d391kg { /* Sidebar background */
        background: rgba(15, 23, 42, 0.4) !important; /* Added !important for specificity */
        backdrop-filter: blur(20px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
    }

    .course-card {
        background-size: cover;
        background-position: center;
        border-radius: 24px;
        padding: 1.5rem; /* Slightly reduced padding for content */
        margin: 0 0 1rem 0; /* Bottom margin for spacing between card and button */
        position: relative;
        overflow: hidden;
        min-height: 380px; 
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
    }

    .course-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.6), rgba(15, 23, 42, 0.7)); /* Adjusted gradient */
        z-index: 1;
    }

    .course-card:hover {
        transform: translateY(-10px) scale(1.01); /* Subtle hover */
    }

    .course-card-content {
        position: relative;
        z-index: 2;
        color: #f8fafc;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .course-card h3 {
        font-size: 1.8rem; /* Slightly smaller */
        font-weight: 700;
        margin-bottom: 0.75rem;
        background: linear-gradient(135deg, #f8fafc, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .course-card p.description {
        font-size: 0.95rem; /* Slightly smaller */
        margin-bottom: 1rem;
        line-height: 1.6;
        color: #e2e8f0;
        display: -webkit-box;
        -webkit-line-clamp: 5; /* Allow more lines for card description */
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
        /* Approximate height for 5 lines, adjust if needed */
        height: calc(1.6em * 5); 
    }

    .course-topics {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0; /* Removed bottom margin as card has space-between */
    }

    .topic-tag {
        background: rgba(59, 130, 246, 0.15);
        color: #a5b4fc; /* Lighter blue */
        padding: 0.3rem 0.7rem;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 500;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white !important; /* Ensure text color */
        border: none;
        border-radius: 16px;
        padding: 0.8rem 1.5rem; /* Adjusted padding */
        font-weight: 600;
        font-size: 0.95rem; /* Adjusted font size */
        transition: all 0.4s ease;
        backdrop-filter: blur(10px);
        width: 100%; /* Ensure button takes full width if use_container_width is true */
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(59, 130, 246, 0.4);
    }
    /* Secondary button style for sidebar (non-active) */
    .stButton button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    .stButton button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: #3b82f6 !important;
        color: #f8fafc !important;
    }


    .chatbot-promo-container {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .chatbot-promo-container h3 {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        font-weight: 700; font-size: 1.8rem;
    }

    .stChatMessage {
        background: rgba(30, 41, 59, 0.5); /* Slightly different background */
        backdrop-filter: blur(15px);
        border-radius: 16px;
        border: 1px solid rgba(71, 85, 105, 0.2); /* Adjusted border */
        margin: 1rem 0;
        padding: 1rem 1.25rem; /* Added padding to chat messages */
    }
    .stChatMessage p { color: #e2e8f0 !important; } /* Ensure text color inside chat messages */


    h1, h2, h3, h4, h5, h6 {
        background: linear-gradient(135deg, #f8fafc, #cbd5e1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        font-weight: 700;
    }
    /* Override for specific headers if they don't pick up global style */
    .stMarkdown h3, .stMarkdown h4, .stMarkdown h2, .stMarkdown h1 {
        background: linear-gradient(135deg, #f8fafc, #cbd5e1) !important;
        -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; background-clip: text !important;
        font-weight: 700 !important;
    }

    .course-list-item {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 1.5rem; /* Adjusted padding */
        margin: 0 0 1.5rem 0; 
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.4s ease;
        min-height: 260px; /* Adjusted min-height */
        display: flex; 
        flex-direction: column; 
        justify-content: space-between; 
    }
    .course-list-item:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);
        border-color: rgba(59, 130, 246, 0.3);
    }
    .course-list-item > div:first-child { flex-grow: 1; }


    .floating-element { animation: elegantFloat 8s ease-in-out infinite; }
    @keyframes elegantFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(2deg); }
    }

    .stat-card {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.3s ease; text-align: center; 
    }
    .stat-card:hover { transform: translateY(-5px); box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3); }

    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.3); border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(135deg, #1d4ed8, #7c3aed); }

    .stMarkdown, .stText, p, div, span, label { color: #e2e8f0; }
    
    .stSelectbox > div > div { 
        background-color: rgba(15, 23, 42, 0.6) !important;
        border-color: rgba(148, 163, 184, 0.2) !important;
    }
    .stSelectbox div[data-baseweb="select"] > div { color: #e2e8f0 !important; }

    .stTextInput > div > div > input { 
        background-color: rgba(15, 23, 42, 0.6) !important;
        border-color: rgba(148, 163, 184, 0.2) !important;
        color: #e2e8f0 !important;
    }
    .stTextInput > label { color: #cbd5e1 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Session State Initialization ---
def initialize_session_state():
    query_params = st.query_params # Use the new st.query_params
    
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = query_params.get("page", "Dashboard")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! What do you want to learn today?"}]
    
    if 'current_dashboard_page' not in st.session_state: # Renamed from current_course_page to be specific
        st.session_state.current_dashboard_page = 0
        
    # Handle direct navigation to course detail via query params
    initial_course_id = query_params.get("course_id", None)
    if initial_course_id:
        if any(course['id'] == initial_course_id for course in COURSES_DATA):
            st.session_state.selected_page = "Course List" # Force page to Course List
            st.session_state.selected_course_for_detail = initial_course_id
        else: # Clear invalid course_id from params
            st.query_params.pop("course_id")


# --- Navigation Function ---
def navigate_to(page_name, course_id=None):
    st.session_state.selected_page = page_name
    
    new_params = {"page": page_name}
    if course_id:
        st.session_state.selected_course_for_detail = course_id
        new_params["course_id"] = course_id
    else:
        if 'selected_course_for_detail' in st.session_state:
            del st.session_state.selected_course_for_detail
        # If navigating away from a specific course detail, remove course_id from params
        if "course_id" in st.query_params:
             st.query_params.pop("course_id") # This might not work as pop is for dicts.
                                              # For st.query_params, setting it to None or empty list
                                              # or simply not including it in set_query_params is the way.
                                              # The current approach is to set only what's needed.
                                              # For complete removal, we'd do st.query_params = {"page": page_name}

    st.query_params.update(new_params) # This will merge. To ensure only these exist, assign directly.
    # For cleaner URL updates:
    current_q_params = st.query_params.to_dict()
    final_params = {}
    if "page" in new_params: final_params["page"] = new_params["page"]
    if "course_id" in new_params: final_params["course_id"] = new_params["course_id"]
    
    # Update only if different to avoid multiple reruns from param setting
    changed_params = False
    for key, value in final_params.items():
        if current_q_params.get(key) != value:
            changed_params = True
            break
    if not changed_params and len(final_params) == len(current_q_params): # no change
        pass
    else:
        st.query_params = final_params


    st.rerun()


# --- Sidebar ---
with st.sidebar:
    st.markdown("<div class='floating-element'>", unsafe_allow_html=True)
    st.title("🎓 EduTech AI") # Shorter title
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    pages = ["Dashboard", "Chatbot", "Course List"]
    for page_name_iter in pages: # Renamed page_name to avoid conflict
        is_active = st.session_state.get('selected_page') == page_name_iter
        button_type = "primary" if is_active else "secondary"
        if st.button(f"{page_name_iter}", use_container_width=True, type=button_type, key=f"nav_{page_name_iter}"):
            # If navigating to Course List from somewhere else, and not to a specific course, clear course_id
            if page_name_iter == "Course List" and 'selected_course_for_detail' in st.session_state:
                 navigate_to(page_name_iter) # Clears course_id by not passing it
            else:
                navigate_to(page_name_iter)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem; background: rgba(15, 23, 42, 0.4); border-radius: 16px; border: 1px solid rgba(148, 163, 184, 0.1);'>
        <p style='font-size: 1rem; color: #cbd5e1; margin: 0;'>
            💡 <strong>AI Assistant</strong><br>
            <span style='font-size: 0.9rem; opacity: 0.8;'>Computer Science Explorer</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- UI Display Functions ---

def display_dashboard():
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(135deg, #f8fafc, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Course AI Explorer
        </h1>
        <p style='font-size: 1.2rem; color: #cbd5e1; max-width: 650px; margin: 0 auto 1.5rem auto; line-height: 1.6;'>
            Exploration of various computer science topics with guidance from AI.
        </p>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="chatbot-promo-container floating-element">
        <div style="display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;">
            <div style="flex-grow: 1;">
                <h3>Need a Study Guide?</h3>
                <p style="margin: 0.75rem 0; font-size: 1rem; color: #cbd5e1; line-height: 1.6;">
                    Consult our AI assistant for topic recommendations and concept explanations.
                </p>
                <div style="display: flex; gap: 0.75rem; margin-top: 1rem; flex-wrap: wrap;">
                    <span style="background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: white; padding: 0.4rem 0.8rem; border-radius: 12px; font-size: 0.85rem; font-weight: 500;">💬 Tanya Seputar CS</span>
                    <span style="background: linear-gradient(135deg, #8b5cf6, #7c3aed); color: white; padding: 0.4rem 0.8rem; border-radius: 12px; font-size: 0.85rem; font-weight: 500;">🎯 Rekomendasi Topik</span>
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    
    if st.button("Start a chat with AI Learning Assistant", type="primary", use_container_width=True, key="dashboard_chat_button"):
        navigate_to("Chatbot")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin: 2rem 0 1rem 0;'><h2 style='font-size: 2.2rem; margin-bottom: 0.5rem;'>Exploration our Course</h2></div>", unsafe_allow_html=True)

    total_courses = len(COURSES_DATA)
    current_page_state = st.session_state.get('current_dashboard_page', 0)
    
    max_page = (total_courses - 1) // COURSES_PER_PAGE_DASHBOARD if total_courses > 0 else 0
    if current_page_state > max_page : current_page_state = max_page
    if current_page_state < 0 : current_page_state = 0
    st.session_state.current_dashboard_page = current_page_state


    start_index = current_page_state * COURSES_PER_PAGE_DASHBOARD
    end_index = min(start_index + COURSES_PER_PAGE_DASHBOARD, total_courses)
    current_courses_to_display = COURSES_DATA[start_index:end_index]

    if not COURSES_DATA:
        st.warning("Belum ada materi yang tersedia saat ini.")
    elif current_courses_to_display:
        num_cols = len(current_courses_to_display)
        cols = st.columns(num_cols if num_cols > 0 else 1)
        for i, course in enumerate(current_courses_to_display):
            with cols[i]:
                card_html = f"""
                <div class="course-card"">
                    <div class="course-card-content">
                        <div>
                            <h3>{course.get('emoji','')} {course['title']}</h3>
                            <p class="description">{course['description']}</p>
                        </div>
                        <div class="course-topics">
                            {''.join([f'<span class="topic-tag">{topic}</span>' for topic in course.get('topics',[])[:3]])}
                        </div>
                    </div>
                </div>"""
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button(f"{course['title']}", key=f"dash_course_{course['id']}", use_container_width=True):
                    navigate_to("Course List", course_id=course['id'])
                    # st.toast(f"✨ Membuka topik {course['title']}") # Toast can be annoying on navigation
    
    if total_courses > COURSES_PER_PAGE_DASHBOARD:
        nav_cols = st.columns([1,1,1]) # Use 3 columns for better spacing
        with nav_cols[0]:
            if st.session_state.current_dashboard_page > 0:
                if st.button("Previous", use_container_width=True, key="prev_dash_course"):
                    st.session_state.current_dashboard_page -= 1
                    st.rerun()
            else: st.markdown("<div style='height:46.4px'></div>", unsafe_allow_html=True) # Placeholder
        with nav_cols[1]:
            total_pages = (total_courses + COURSES_PER_PAGE_DASHBOARD - 1) // COURSES_PER_PAGE_DASHBOARD
            st.markdown(f"<div style='text-align: center; padding-top:0.75rem; font-size: 1rem; color: #cbd5e1;'>Page {st.session_state.current_dashboard_page + 1} dari {total_pages}</div>", unsafe_allow_html=True)
        with nav_cols[2]:
            if end_index < total_courses:
                if st.button("Next", use_container_width=True, key="next_dash_course"):
                    st.session_state.current_dashboard_page += 1
                    st.rerun()
            else: st.markdown("<div style='height:46.4px'></div>", unsafe_allow_html=True) # Placeholder
    
    st.markdown("---")
    if st.button("See all the course list", use_container_width=True, key="dashboard_all_courses"):
        navigate_to("Course List")

chatbot_instance = get_chatbot_instance()

def display_chatbot():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    chatbot_instance = get_chatbot_instance()
    if not chatbot_instance:
        st.error("Chatbot tidak dapat diinisialisasi. Mohon periksa konfigurasi.")
        return

    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'><h1 style='font-size: 2.5rem;'>AI Learning Assistant</h1></div>", unsafe_allow_html=True)
    
    # Quick Action Buttons (Optional - can be removed if bot is robust)
    # st.markdown("### 💡 Pertanyaan Populer:") ... 

    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True) # Allow HTML for links

    if prompt := st.chat_input("Ask about Computer Science", key="chatbot_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("AI is thinking..."):
            bot_response_text, contexts, _ = chatbot_instance.generate_response(prompt)
        
        assistant_message_content = bot_response_text

        # Add link if relevant context found and it's a course ID
        if contexts and contexts[0]['similarity'] > 0.7: # Adjust threshold as needed
            relevant_course_id = contexts[0]['tag']
            # Check if this tag is a valid course ID from our loaded courses
            matching_course = next((c for c in COURSES_DATA if c['id'] == relevant_course_id), None)
            if matching_course:
                course_title = matching_course['title']
                # Construct URL with query parameters. Page name needs to be URL encoded if it has spaces.
                # Streamlit handles this automatically if we pass dict to st.query_params
                # For markdown link, manually create the query string part
                encoded_relevant_course_id = urllib.parse.quote_plus(relevant_course_id)

                link_query_params = f"page=Course+List&course_id={encoded_relevant_course_id}" # Menggunakan ID yang sudah di-encode
                
                # Streamlit base URL is handled by browser, so relative link is fine
                # For links in markdown to trigger st.query_params, they might need to be full or relative path
                # A simple query string like "?page=...&course_id=..." works.
                link_markdown = f"\n\nTo learn more, you can view the topic here: [**{course_title}**](?{link_query_params})"
                assistant_message_content += link_markdown
        
        st.session_state.messages.append({"role": "assistant", "content": assistant_message_content})
        st.rerun()


def display_course_list():
    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'><h1 style='font-size: 2.5rem;'>Course List</h1></div>", unsafe_allow_html=True)

    if st.session_state.get('selected_course_for_detail'):
        show_course_detail(st.session_state.selected_course_for_detail)
        return

    search_query = st.text_input("🔍  Search any CS topic", placeholder="Example: abstraction, python, error handling", key="course_search")
    
    stats_cols = st.columns(3) # Simplified stats
    total_topics_count = len(COURSES_DATA)
    # Assuming 'topics' in COURSES_DATA are the sub-topics for this count
    total_sub_topics_count = sum(len(course.get("topics", [])) for course in COURSES_DATA) 

    with stats_cols[0]: st.markdown(f"<div class='stat-card'><div style='font-size:1.5rem;font-weight:600;color:#3b82f6;'>{total_topics_count}</div><div>Topics</div></div>", unsafe_allow_html=True)
    with stats_cols[1]: st.markdown(f"<div class='stat-card'><div style='font-size:1.5rem;font-weight:600;color:#10b981;'>{total_sub_topics_count}</div><div>Sub-Topics</div></div>", unsafe_allow_html=True)
    with stats_cols[2]: st.markdown("<div class='stat-card'><div style='font-size:1.5rem;font-weight:600;color:#f59e0b;'>AI Powered</div><div>Learning</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    filtered_courses = COURSES_DATA
    if search_query:
        sq_lower = search_query.lower()
        filtered_courses = [
            course for course in COURSES_DATA 
            if sq_lower in course['title'].lower() or 
               sq_lower in course['description'].lower() or
               any(sq_lower in topic.lower() for topic in course.get('topics',[]))
        ]

    if not COURSES_DATA:
        st.info("Katalog topik sedang disiapkan. Silakan cek kembali nanti.")
    elif not filtered_courses:
        st.markdown("<div style='text-align:center;padding:2rem;background:rgba(15,23,42,0.3);border-radius:15px;'><div style='font-size:3rem;'>🚫</div><h3>Topik tidak ditemukan</h3><p>Coba kata kunci lain.</p></div>", unsafe_allow_html=True)
    else:
        for i in range(0, len(filtered_courses), 2):
            cols = st.columns(2)
            for j, col_widget in enumerate(cols):
                if i + j < len(filtered_courses):
                    course = filtered_courses[i + j]
                    with col_widget:
                        display_course_card_item(course) # Renamed for clarity

def display_course_card_item(course): # Was display_course_card_detailed
    """Display individual course card in the list view."""
    description_style = (
        "font-size: 0.9rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 1rem; "
        "height: calc(1.5em * 4); display: -webkit-box; -webkit-line-clamp: 4; " # 4 lines
        "-webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis;"
    )
    card_html = f"""
    <div class="course-list-item"> 
        <div style="display: flex; gap: 1rem; align-items: flex-start; flex-grow: 1;"> 
            <div style="flex: 1;">
                <h4 style="font-size: 1.3rem; margin-bottom: 0.5rem; color: #f0f0f0; background: none; -webkit-text-fill-color: unset;">{course['title']}</h4>
                <p style="{description_style}">{course['description']}</p>
                <div class="course-topics" style="margin-bottom:0;">{''.join([f'<span class="topic-tag">{topic}</span>' for topic in course.get('topics',[])[:4]])}</div>
            </div>
        </div>
    </div>"""
    st.markdown(card_html, unsafe_allow_html=True)
    if st.button(f"See the detail: {course['title']}", key=f"learn_{course['id']}", use_container_width=True):
        navigate_to("Course List", course_id=course['id'])


def show_course_detail(course_id):
    course_meta = next((c for c in COURSES_DATA if c['id'] == course_id), None)
    course_content = COURSE_CONTENT_DETAILS.get(course_id)

    if not course_meta or not course_content:
        st.error("Detail topik tidak ditemukan.")
        if st.button("Back to the Course List", key="detail_back_err", use_container_width=True):
            navigate_to("Course List")
        return

    # st.markdown(f"""
    # <div style='background:linear-gradient(135deg, rgba(59,130,246,0.1), rgba(139,92,246,0.1)); padding:2rem; border-radius:20px; margin:1rem 0 2rem 0; text-align:center; border:1px solid rgba(148,163,184,0.15);'>
    #     <h1 style='font-size:2.5rem; margin-bottom:0.75rem; color:#f0f0f0;'>{course_meta['title']}</h1>
    #     <p style='font-size:1.1rem; color:#cbd5e1; line-height:1.6; max-width:750px; margin:0 auto;'>{course_meta['description']}</p>
    # </div>""", unsafe_allow_html=True)

    if course_content.get('overview'):
        st.write("  ")
        st.markdown("###  About the Lesson")
        # Perform the replacement operations outside the f-string
        overview_html = course_content['overview']
        st.markdown(f"<div style='background:rgba(15,23,42,0.35); padding:1.5rem; border-radius:15px; margin:1rem 0; border:1px solid rgba(148,163,184,0.1); backdrop-filter:blur(10px);'><p style='font-size:1rem; color:#e2e8f0; line-height:1.7; text-align:justify;'>{overview_html}</p></div>", unsafe_allow_html=True)

    if course_content.get('key_concepts'):
        st.markdown("### Main Concept")
        for i, concept in enumerate(course_content['key_concepts']):
            st.markdown(f"<div style='background:rgba(15,23,42,0.25); padding:1rem 1.25rem; border-radius:12px; margin:0.5rem 0; border-left:3px solid #3b82f6;'><h5 style='color:white; margin-bottom:0.3rem; font-size:1.1rem; background:none; -webkit-text-fill-color:unset;'>{i+1}. {concept.split(': ')[0]}</h5><p style='color:#cbd5e1; font-size:1rem; line-height:1.6; margin:0;'>{': '.join(concept.split(': ')[1:]) if ': ' in concept else ''}</p></div>", unsafe_allow_html=True)
            
    if course_content.get('code_example') and course_content['code_example'].get('code'):
        st.markdown("### Example code")
        lang = course_content['code_example'].get('language', 'plaintext')
        code = course_content['code_example'].get('code', '')
        st.code(code, language=lang, line_numbers=True)

    if course_content.get('applications'):
        st.markdown("### Realworld Application")
        for app in course_content['applications']:
            st.markdown(f"<div style='background:rgba(16,185,129,0.08); padding:0.8rem 1rem; border-radius:10px; margin:0.5rem 0; border:1px solid rgba(16,185,129,0.15); display:flex; align-items:center; gap:0.75rem;'><p style='color:#e2e8f0; margin:0; font-size:0.95rem;'>{app}</p></div>", unsafe_allow_html=True)

    st.markdown(f"<br><br>", unsafe_allow_html=True)
    if st.button("Back to Course List", key="back_to_list_detail", use_container_width=True):
        navigate_to("Course List")


# --- Main App Logic ---
def main():
    load_and_parse_courses_from_json() # Load data at the start
    initialize_session_state() # Initialize/update session state based on query_params AFTER data load

    # Display selected page
    page_to_display = st.session_state.get('selected_page', "Dashboard")

    if page_to_display == "Dashboard":
        display_dashboard()
    elif page_to_display == "Chatbot":
        display_chatbot()
    elif page_to_display == "Course List":
        display_course_list()
    else: # Fallback to Dashboard if page unknown
        st.session_state.selected_page = "Dashboard"
        display_dashboard()

if __name__ == "__main__":
    
    main()
