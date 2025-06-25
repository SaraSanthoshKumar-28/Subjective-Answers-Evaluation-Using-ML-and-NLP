import streamlit as st
import sqlite3
import bcrypt
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import language_tool_python
import nltk
import string
from nltk.corpus import wordnet, stopwords
from datetime import datetime
import io
import re
import json
import logging
from functools import lru_cache
import csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(page_title="Subjective Answer Evaluator", layout="wide", initial_sidebar_state="auto")

# NLTK downloads
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Custom CSS
st.markdown("""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
@import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');

body, .stApp {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
    margin: 0 !important;
    padding: 0 !important;
}

h1, h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
}

.stTextInput input, .stTextArea textarea {
    border: 2px solid #3b82f6 !important;
    border-radius: 0.5rem !important;
    padding: 0.75rem !important;
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #60a5fa !important;
    background-color: #1e293b !important;
    box-shadow: 0 0 10px rgba(96, 165, 250, 0.3) !important;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 0.5rem !important;
}

.stButton button {
    background-color: #3b82f6 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 0.5rem !important;
    padding: 0.75rem 1.5rem !important;
}

.stButton button:hover {
    background-color: #2563eb !important;
    transform: scale(1.05) !important;
}

.stExpander {
    background-color: #1e293b !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 0.5rem !important;
}

.stDataFrame {
    background-color: #1e293b !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 0.5rem !important;
}

.stSidebar {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
}

.stSidebar .stButton button {
    background-color: #3b82f6 !important;
    color: #ffffff !important;
}

.stSidebar .stButton button:hover {
    background-color: #2563eb !important;
}

.css-1aumxhk {
    margin-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Database context manager
class Database:
    def __init__(self, db_name='evaluator.db'):
        self.db_name = db_name

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name, timeout=10)
        self.c = self.conn.cursor()
        self.c.execute('PRAGMA foreign_keys = ON')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.conn.close()

# Database setup with migration
def init_db():
    with Database() as db:
        try:
            # Check if users table exists and add subjects column if missing
            db.c.execute('PRAGMA table_info(users)')
            columns = [info[1] for info in db.c.fetchall()]
            if 'subjects' not in columns:
                db.c.execute('ALTER TABLE users ADD COLUMN subjects TEXT')
                logger.info("Added 'subjects' column to users table.")

            db.c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'teacher', 'student')),
                    subjects TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    is_approved BOOLEAN NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            db.c.execute('''
                CREATE TABLE IF NOT EXISTS assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    teacher_id INTEGER,
                    subject TEXT NOT NULL,
                    question TEXT NOT NULL,
                    model_answer TEXT NOT NULL,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(teacher_id) REFERENCES users(id) ON DELETE CASCADE
                )
            ''')
            db.c.execute('''
                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    assignment_id INTEGER,
                    answer TEXT NOT NULL,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'Pending',
                    evaluation_result TEXT,
                    FOREIGN KEY(student_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY(assignment_id) REFERENCES assignments(id) ON DELETE CASCADE
                )
            ''')
            admin_username = 'Akshith'
            db.c.execute('SELECT id FROM users WHERE username = ?', (admin_username,))
            if not db.c.fetchone():
                hashed_password = bcrypt.hashpw('admin123'.encode(), bcrypt.gensalt()).decode()
                db.c.execute('INSERT INTO users (username, password, role, is_active, is_approved) VALUES (?, ?, ?, ?, ?)',
                             (admin_username, hashed_password, 'admin', 1, 1))
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            logger.error(f"Database error: {e}")

init_db()

# Password handling
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception as e:
        logger.error(f"Password error: {e}")
        return False

# Authentication
def authenticate(username, password):
    with Database() as db:
        db.c.execute('SELECT id, username, password, role, is_approved, subjects FROM users WHERE username = ? AND is_active = 1', (username,))
        user = db.c.fetchone()
        if user and verify_password(password, user[2]):
            if user[3] != 'admin' and not user[4]:
                return None, "Account not approved."
            return {'id': user[0], 'username': user[1], 'role': user[3], 'subjects': user[5] or ''}, None
        return None, "Invalid credentials."

# User management
def manage_user(action, user_id):
    with Database() as db:
        try:
            if action == 'remove':
                db.c.execute('DELETE FROM users WHERE id = ?', (user_id,))
                return "User removed."
            elif action == 'inactivate':
                db.c.execute('UPDATE users SET is_active = 0 WHERE id = ?', (user_id,))
                return "User inactivated."
            elif action == 'activate':
                db.c.execute('UPDATE users SET is_active = 1 WHERE id = ?', (user_id,))
                return "User activated."
        except sqlite3.Error as e:
            return f"Error: {e}"
    return "Invalid action."

# Model and tools
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_language_tool():
    return language_tool_python.LanguageTool('en-US')

# Text processing
def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    return " ".join([word for word in tokens if word not in stopwords.words('english')])

@st.cache_data
def encode_answer(model, text):
    return model.encode(text, convert_to_tensor=True)

def semantic_similarity(ans1, ans2):
    if not ans1 or not ans2:
        return 0.0, 0.0
    model = load_model()
    embedding1 = encode_answer(model, ans1)
    embedding2 = encode_answer(model, ans2)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return round(float(similarity[0][0]) * 100, 2), 0.0

def score_similarity(similarity):
    if similarity < 10:
        return 0, "Off-topic"
    elif similarity < 30:
        return 2, "Very vague"
    elif similarity < 50:
        return 4, "Partially relevant"
    elif similarity < 70:
        return 6, "Moderately relevant"
    elif similarity < 85:
        return 8, "Good relevance"
    else:
        return 10, "Excellent relevance"

def grammar_score(text):
    if not text or not text.strip():
        return 0, "Empty", []
    tool = load_language_tool()
    words = len(text.split())
    matches = tool.check(text)
    errors = len(matches)
    error_rate = min(errors / words, 0.5) if words > 0 else 0.5
    if errors == 0:
        return 10, "Perfect", matches
    elif error_rate < 0.02:
        return 8, "Minor issues", matches
    elif error_rate < 0.05:
        return 6, "Noticeable errors", matches
    else:
        return 2, "Poor", matches

def length_penalty(text):
    if not text or not text.strip():
        return 0, "Empty"
    words = len(text.split())
    if words < 15:
        return 0.5, "Too short"
    elif words > 200:
        return 0.8, "Too long"
    return 1.0, "Appropriate"

def keyword_match(student, keywords):
    if not student or not keywords:
        return 0, "No data", []
    student_words = set(preprocess(student).split())
    matched = []
    for kw in keywords:
        if any(syn in student_words for syn in get_synonyms(kw)):
            matched.append(kw)
    match_rate = len(matched) / len(keywords) if keywords else 0
    if match_rate >= 0.5:
        return 10, "Good coverage", matched
    elif match_rate >= 0.3:
        return 4, "Partial coverage", matched
    else:
        return 0, "Significant mismatch", matched

def get_synonyms(word):
    synonyms = {word}
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def evaluate_answer(model_answer, student_answer, keywords):
    if not model_answer or not student_answer:
        return {
            'Similarity (%)': 0.0, 'Similarity Score': 0, 'Similarity Feedback': 'No content',
            'Grammar Score': 0, 'Grammar Feedback': 'No content', 'Grammar Matches': [],
            'Keyword Score': 0, 'Keyword Feedback': 'No content', 'Matched Keywords': 'None',
            'Length Penalty': 0, 'Length Feedback': 'No content', 'Final Score': 0.0,
            'Suggestions': 'Provide answers.'
        }
    similarity, _ = semantic_similarity(model_answer, student_answer)
    sim_score, sim_feedback = score_similarity(similarity)
    if sim_score == 0:
        return {
            'Similarity (%)': similarity, 'Similarity Score': 0, 'Similarity Feedback': sim_feedback,
            'Grammar Score': 0, 'Grammar Feedback': 'Not evaluated', 'Grammar Matches': [],
            'Keyword Score': 0, 'Keyword Feedback': 'Not evaluated', 'Matched Keywords': 'None',
            'Length Penalty': 0, 'Length Feedback': 'Not evaluated', 'Final Score': 0.0,
            'Suggestions': 'Answer is off-topic.'
        }
    grammar_score_val, grammar_feedback, grammar_matches = grammar_score(student_answer)
    keyword_score, keyword_feedback, matched_keywords = keyword_match(student_answer, keywords)
    length_factor, length_feedback = length_penalty(student_answer)
    weights = {'similarity': 0.4, 'keywords': 0.4, 'grammar': 0.1, 'length': 0.1}
    raw_score = (sim_score * weights['similarity'] + keyword_score * weights['keywords'] +
                 grammar_score_val * weights['grammar']) * length_factor
    final_score = round(min(10.0, max(0.0, raw_score)), 1)
    suggestions = []
    if sim_score < 6:
        suggestions.append("Improve relevance.")
    if keyword_score < 6:
        suggestions.append("Include key concepts.")
    if grammar_score_val < 6:
        suggestions.append("Check grammar.")
    if length_factor < 1.0:
        suggestions.append("Expand answer.")
    return {
        'Similarity (%)': similarity, 'Similarity Score': sim_score, 'Similarity Feedback': sim_feedback,
        'Grammar Score': grammar_score_val, 'Grammar Feedback': grammar_feedback, 'Grammar Matches': grammar_matches,
        'Keyword Score': keyword_score, 'Keyword Feedback': keyword_feedback, 'Matched Keywords': ', '.join(matched_keywords) or 'None',
        'Length Penalty': length_factor, 'Length Feedback': length_feedback, 'Final Score': final_score,
        'Suggestions': ' '.join(suggestions) if suggestions else 'Well done!'
    }

# Registration
def register_user(username, password, role, subjects=None):
    if len(username) < 4 or len(password) < 6:
        return False, "Username ≥4 chars, password ≥6 chars."
    if role not in ['teacher', 'student']:
        return False, "Invalid role."
    if role == 'teacher' and not subjects:
        return False, "Please select at least one subject for a teacher."
    with Database() as db:
        try:
            subjects_str = subjects if subjects else ''
            logger.info(f"Registering {username} as {role} with subjects: {subjects_str}")
            db.c.execute('INSERT INTO users (username, password, role, subjects, is_active, is_approved) VALUES (?, ?, ?, ?, ?, ?)',
                         (username, hash_password(password), role.lower(), subjects_str, 1, 0))
            return True, f"Registration successful for {role}! Awaiting approval."
        except sqlite3.IntegrityError:
            return False, "Username exists."
        except sqlite3.Error as e:
            return False, f"Error: {e}"

# User management
def manage_users():
    st.subheader("User Management")
    with Database() as db:
        db.c.execute('SELECT id, username, role, subjects, is_active, is_approved FROM users WHERE role != "admin"')
        users = db.c.fetchall()
        if users:
            users_df = pd.DataFrame(users, columns=['ID', 'Username', 'Role', 'Subjects', 'Active', 'Approved'])
            st.dataframe(users_df, use_container_width=True)
            user_id = st.number_input("User ID", min_value=1, step=1)
            action = st.selectbox("Action", ["Remove", "Inactivate", "Activate"])
            if st.button("Execute"):
                result = manage_user(action.lower(), user_id)
                st.success(result)
                st.rerun()
        else:
            st.info("No users.")

    st.subheader("Pending Approvals")
    with Database() as db:
        db.c.execute('SELECT id, username, role, subjects, created_at FROM users WHERE is_approved = 0 AND is_active = 1')
        pending = db.c.fetchall()
        if pending:
            pending_df = pd.DataFrame(pending, columns=['ID', 'Username', 'Role', 'Subjects', 'Registered On'])
            st.dataframe(pending_df, use_container_width=True)
            user_id = st.number_input("Approve User ID", min_value=1, step=1)
            if st.button("Approve"):
                db.c.execute('UPDATE users SET is_approved = 1 WHERE id = ?', (user_id,))
                st.success("Approved!")
                st.rerun()
        else:
            st.info("No pending approvals.")

# Assignments and submissions
def get_assignments(user_id, role):
    with Database() as db:
        try:
            if role == 'student':
                db.c.execute('''
                    SELECT a.id, a.subject, a.question, a.created_at, u.username,
                           (SELECT COUNT(*) FROM submissions s WHERE s.assignment_id = a.id AND s.student_id = ?) as submitted
                    FROM assignments a JOIN users u ON a.teacher_id = u.id
                    WHERE u.subjects LIKE ? OR u.subjects = ''
                ''', (user_id, f'%{role}%'))
            elif role == 'teacher':
                db.c.execute('''
                    SELECT a.id, a.subject, a.question, a.created_at, u.username,
                           (SELECT COUNT(*) FROM submissions s WHERE s.assignment_id = a.id) as submitted
                    FROM assignments a JOIN users u ON a.teacher_id = u.id
                    WHERE a.teacher_id = ?
                ''', (user_id,))
            else:  # admin
                db.c.execute('''
                    SELECT a.id, a.subject, a.question, a.created_at, u.username,
                           (SELECT COUNT(*) FROM submissions s WHERE s.assignment_id = a.id) as submitted
                    FROM assignments a JOIN users u ON a.teacher_id = u.id
                ''')
            return db.c.fetchall()
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            return []

def get_submissions(user_id, role):
    with Database() as db:
        try:
            if role == 'student':
                db.c.execute('''
                    SELECT s.id, a.subject, a.question, s.answer, s.submitted_at, s.status, s.evaluation_result, u.username
                    FROM submissions s JOIN assignments a ON s.assignment_id = a.id
                    JOIN users u ON a.teacher_id = u.id WHERE s.student_id = ?
                ''', (user_id,))
            elif role == 'teacher':
                db.c.execute('''
                    SELECT s.id, a.subject, a.question, s.answer, s.submitted_at, s.status, s.evaluation_result, u.username
                    FROM submissions s JOIN assignments a ON s.assignment_id = a.id
                    JOIN users u ON s.student_id = u.id WHERE a.teacher_id = ?
                ''', (user_id,))
            else:
                db.c.execute('''
                    SELECT s.id, a.subject, a.question, s.answer, s.submitted_at, s.status, s.evaluation_result, u.username
                    FROM submissions s JOIN assignments a ON s.assignment_id = a.id
                    JOIN users u ON s.student_id = u.id
                ''')
            return db.c.fetchall()
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            return []

def has_submitted(student_id, assignment_id):
    with Database() as db:
        db.c.execute('SELECT COUNT(*) FROM submissions WHERE student_id = ? AND assignment_id = ?', (student_id, assignment_id))
        return db.c.fetchone()[0] > 0

def get_teacher_id_for_subject(subject):
    with Database() as db:
        db.c.execute('SELECT id FROM users WHERE role = "teacher" AND (subjects LIKE ? OR subjects = "") LIMIT 1', (f'%{subject}%',))
        result = db.c.fetchone()
        return result[0] if result else None

# Main app
if 'user' not in st.session_state:
    st.session_state.user = None
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'auth_action' not in st.session_state:
    st.session_state.auth_action = "Login"
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'password' not in st.session_state:
    st.session_state.password = ""

st.title("📘 Subjective Answer Evaluator")

if not st.session_state.user:
    with st.container():
        auth_action = st.radio("Select Action", ["Login", "Register"], key="auth_action_radio", on_change=lambda: st.session_state.update({"username": "", "password": ""}))
        if auth_action != st.session_state.auth_action:
            st.session_state.auth_action = auth_action
            st.rerun()

        with st.form("auth_form"):
            st.session_state.username = st.text_input("Username", value=st.session_state.username, key=f"auth_{auth_action}_username")
            st.session_state.password = st.text_input("Password", type="password", value=st.session_state.password, key=f"auth_{auth_action}_password")
            if auth_action == "Register":
                role = st.selectbox("Role", ["Teacher", "Student"], key="register_role")
                subjects = st.multiselect("Subjects" if role == "Teacher" else "Subjects of Interest", ["Biology", "Physics", "History", "Literature", "Mathematics", "Chemistry", "Geometry", "Other"], key=f"register_{role.lower()}_subjects", default=None) if role in ["Teacher", "Student"] else []
                subjects_str = ",".join(subjects) if subjects else None
                logger.info(f"Role: {role}, Subjects: {subjects_str}")
            submitted = st.form_submit_button("Submit")
            if submitted:
                if not st.session_state.username or not st.session_state.password:
                    st.error("Enter username and password.")
                elif auth_action == "Login":
                    user, error = authenticate(st.session_state.username, st.session_state.password)
                    if user:
                        st.session_state.user = user
                        st.success(f"Welcome, {st.session_state.username}!")
                        st.rerun()
                    else:
                        st.error(error)
                else:
                    success, message = register_user(st.session_state.username, st.session_state.password, role, subjects_str)
                    if success:
                        st.success(message)
                        st.session_state.username = ""
                        st.session_state.password = ""
                    else:
                        st.error(message)
else:
    user = st.session_state.user
    st.sidebar.title(f"👤 {user['username']} ({user['role'].capitalize()})")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.username = ""
        st.session_state.password = ""
        st.session_state.notifications = []
        st.session_state.auth_action = "Login"
        st.rerun()

    if st.session_state.notifications:
        with st.sidebar.expander("🔔 Notifications"):
            for notif in st.session_state.notifications:
                st.write(notif)
        if st.sidebar.button("Clear"):
            st.session_state.notifications = []
            st.rerun()

    if user['role'] == 'admin':
        st.header("📘 Admin Dashboard")
        manage_users()

        st.subheader("Assignments")
        assignments = get_assignments(user['id'], user['role'])
        if assignments:
            assignments_df = pd.DataFrame(assignments, columns=['ID', 'Subject', 'Question', 'Created At', 'Teacher', 'Submissions'])
            st.dataframe(assignments_df)
        else:
            st.info("No assignments.")

        with st.expander("Evaluate Submissions"):
            submissions = get_submissions(user['id'], user['role'])
            pending = [(s[0], s[1], s[2], s[3], s[4], s[7]) for s in submissions if s[5] == 'Pending']
            if pending:
                submissions_df = pd.DataFrame(pending, columns=['ID', 'Subject', 'Question', 'Answer', 'Submitted At', 'Student'])
                st.dataframe(submissions_df)
                submission_id = st.number_input("Submission ID", min_value=1)
                if submission_id:
                    with Database() as db:
                        db.c.execute('SELECT s.answer, a.question, a.model_answer, a.keywords FROM submissions s JOIN assignments a ON s.assignment_id = a.id WHERE s.id = ? AND s.status = ?', (submission_id, 'Pending'))
                        submission = db.c.fetchone()
                        if submission:
                            student_answer, question, model_answer, keywords_str = submission
                            keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
                            st.text_area("Question", value=question, height=100, disabled=True)
                            st.text_area("Student's Answer", value=student_answer, height=150, disabled=True)
                            st.text_area("Model Answer", value=model_answer, height=150, disabled=True)
                            st.text_input("Keywords", value=", ".join(keywords), disabled=True)
                            if st.button("Evaluate"):
                                with st.spinner("Evaluating..."):
                                    result = evaluate_answer(model_answer, student_answer, keywords)
                                    evaluation_result = json.dumps(result)
                                    db.c.execute('UPDATE submissions SET status = ?, evaluation_result = ? WHERE id = ?', ("Evaluated", evaluation_result, submission_id))
                                    st.success("Complete")
                                    st.session_state.notifications.append(f"Submission {submission_id} evaluated at {datetime.now().strftime('%H:%M:%S')}.")
                                    st.markdown("### Evaluation")
                                    with st.container():
                                        st.write(f"Similarity: {result['Similarity (%)']:.1f}% ({result['Similarity Score']}/10) - {result['Similarity Feedback']}")
                                        st.write(f"Grammar: {result['Grammar Score']}/10 - {result['Grammar Feedback']}")
                                        st.write(f"Keywords: {result['Keyword Score']}/10 - {result['Keyword Feedback']} (Matched: {result['Matched Keywords']})")
                                        st.write(f"Length: {result['Length Feedback']}")
                                        st.write(f"Final Score: {result['Final Score']}/10")
                                        if result['Suggestions']:
                                            st.markdown("#### Suggestions")
                                            st.write(result['Suggestions'])
                        else:
                            st.error("Invalid submission.")
            else:
                st.info("No pending submissions.")

        st.subheader("Export Data")
        with Database() as db:
            db.c.execute('SELECT * FROM users')
            users_df = pd.DataFrame(db.c.fetchall(), columns=['ID', 'Username', 'Password', 'Role', 'Subjects', 'Active', 'Approved', 'Created At'])
            db.c.execute('SELECT * FROM submissions')
            submissions_df = pd.DataFrame(db.c.fetchall(), columns=['ID', 'Student ID', 'Assignment ID', 'Answer', 'Submitted At', 'Status', 'Evaluation Result'])
            csv_buffer = io.StringIO()
            pd.concat([users_df, submissions_df]).to_csv(csv_buffer, index=False)
            st.download_button("Download CSV", csv_buffer.getvalue(), "data_export.csv", "text/csv")

    elif user['role'] == 'teacher':
        st.header("📘 Teacher Dashboard")
        with st.expander("Create Assignment"):
            subject = st.selectbox("Subject", ["Biology", "Physics", "History", "Literature", "Mathematics", "Chemistry", "Geometry", "Other"])
            question = st.text_area("Question (max 1000 chars)", max_chars=1000).strip()
            model_answer = st.text_area("Model Answer (max 1000 chars)", max_chars=1000).strip()
            keywords = st.text_input("Keywords (comma-separated)").strip().split(",")
            with st.expander("Model Examples"):
                if subject == "Biology":
                    st.write("Example: Photosynthesis converts sunlight into energy using chlorophyll.")
                elif subject == "Physics":
                    st.write("Example: Newton's first law states an object at rest stays at rest.")
                elif subject == "History":
                    st.write("Example: The French Revolution began in 1789 due to economic crises.")
                elif subject == "Literature":
                    st.write("Example: In 'Frankenstein', isolation drives the Creature to despair.")
                elif subject == "Mathematics":
                    st.write("Example: Solve 2x + 3 = 7 to find x = 2.")
                elif subject == "Chemistry":
                    st.write("Example: Water (H₂O) is formed by two hydrogen atoms and one oxygen atom.")
                elif subject == "Geometry":
                    st.write("Example: A triangle has three sides and the sum of angles is 180°.")
                else:
                    st.write("Example: Provide a general overview of the topic.")
            if st.button("Create"):
                if not question or not model_answer:
                    st.error("Required fields missing.")
                else:
                    with Database() as db:
                        try:
                            db.c.execute('INSERT INTO assignments (teacher_id, subject, question, model_answer, keywords) VALUES (?, ?, ?, ?, ?)',
                                         (user['id'], subject, question, model_answer, ",".join(keywords)))
                            st.success("Created!")
                            st.rerun()
                        except sqlite3.Error as e:
                            st.error(f"Error: {e}")

        with st.expander("Evaluate Submissions"):
            submissions = get_submissions(user['id'], user['role'])
            pending = [(s[0], s[1], s[2], s[3], s[4], s[7]) for s in submissions if s[5] == 'Pending']
            if pending:
                submissions_df = pd.DataFrame(pending, columns=['ID', 'Subject', 'Question', 'Answer', 'Submitted At', 'Student'])
                st.dataframe(submissions_df)
                submission_id = st.number_input("Submission ID", min_value=1)
                if submission_id:
                    with Database() as db:
                        db.c.execute('SELECT s.answer, a.question, a.model_answer, a.keywords FROM submissions s JOIN assignments a ON s.assignment_id = a.id WHERE s.id = ? AND s.status = ?', (submission_id, 'Pending'))
                        submission = db.c.fetchone()
                        if submission:
                            student_answer, question, model_answer, keywords_str = submission
                            keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
                            st.text_area("Question", value=question, height=100, disabled=True)
                            st.text_area("Student's Answer", value=student_answer, height=150, disabled=True)
                            st.text_area("Model Answer", value=model_answer, height=150, disabled=True)
                            st.text_input("Keywords", value=", ".join(keywords), disabled=True)
                            if st.button("Evaluate"):
                                with st.spinner("Evaluating..."):
                                    result = evaluate_answer(model_answer, student_answer, keywords)
                                    evaluation_result = json.dumps(result)
                                    db.c.execute('UPDATE submissions SET status = ?, evaluation_result = ? WHERE id = ?', ("Evaluated", evaluation_result, submission_id))
                                    st.success("Complete")
                                    st.session_state.notifications.append(f"Submission {submission_id} evaluated at {datetime.now().strftime('%H:%M:%S')}.")
                                    st.markdown("### Evaluation")
                                    with st.container():
                                        st.write(f"Similarity: {result['Similarity (%)']:.1f}% ({result['Similarity Score']}/10) - {result['Similarity Feedback']}")
                                        st.write(f"Grammar: {result['Grammar Score']}/10 - {result['Grammar Feedback']}")
                                        st.write(f"Keywords: {result['Keyword Score']}/10 - {result['Keyword Feedback']} (Matched: {result['Matched Keywords']})")
                                        st.write(f"Length: {result['Length Feedback']}")
                                        st.write(f"Final Score: {result['Final Score']}/10")
                                        if result['Suggestions']:
                                            st.markdown("#### Suggestions")
                                            st.write(result['Suggestions'])
                        else:
                            st.error("Invalid submission.")
            else:
                st.info("No pending submissions.")

    else:  # Student
        st.header("📘 Student Dashboard")
        assignments = get_assignments(user['id'], user['role'])
        pending = [(a[0], a[1], a[2], a[3], a[4]) for a in assignments if a[5] == 0]
        submitted = [(a[0], a[1], a[2], a[3], a[4]) for a in assignments if a[5] > 0]
        if pending:
            st.subheader("Pending Assignments")
            st.dataframe(pd.DataFrame(pending, columns=['ID', 'Subject', 'Question', 'Created At', 'Teacher']))
            assignment_id = st.number_input("Assignment ID", min_value=1)
            if assignment_id:
                with Database() as db:
                    db.c.execute('SELECT subject, question FROM assignments WHERE id = ?', (assignment_id,))
                    result = db.c.fetchone()
                    if result and not has_submitted(user['id'], assignment_id):
                        subject, question = result
                        teacher_id = get_teacher_id_for_subject(subject)
                        if teacher_id:
                            st.text_area("Question", value=question, height=100, disabled=True)
                            student_answer = st.text_area("Your Answer (max 1000 chars)", max_chars=1000).strip()
                            if st.button("Submit"):
                                if student_answer:
                                    db.c.execute('INSERT INTO submissions (student_id, assignment_id, answer, status) VALUES (?, ?, ?, ?)',
                                                 (user['id'], assignment_id, student_answer, "Pending"))
                                    st.success("Submitted!")
                                    st.rerun()
                                else:
                                    st.error("Answer required.")
                        else:
                            st.error("No teacher for this subject.")
                    elif result:
                        st.error("Already submitted.")
                    else:
                        st.error("Invalid assignment.")
        else:
            st.info("No pending assignments.")
        if submitted:
            st.subheader("Submitted Assignments")
            st.dataframe(pd.DataFrame(submitted, columns=['ID', 'Subject', 'Question', 'Created At', 'Teacher']))

    with st.expander("Evaluation History"):
        submissions = get_submissions(user['id'], user['role'])
        if submissions:
            history = [{'ID': s[0], 'Subject': s[1], 'Question': s[2][:50] + "...", 'Answer': s[3][:50] + "...", 'Submitted At': s[4], 'Status': s[5], 'Student': s[7]} for s in submissions]
            for h in history:
                result = json.loads(next(s for s in submissions if s[0] == h['ID'])[6]) if any(s[6] for s in submissions) else {}
                h.update({'Score': f"{result.get('Final Score', 0.0)}/10"})
            st.dataframe(pd.DataFrame(history))
        else:
            st.info("No history.")

st.markdown("---")
st.markdown("Made with ❤️ by Akshith")