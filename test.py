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
import datetime as dt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(page_title="Subjective Answer Evaluator", layout="wide", initial_sidebar_state="auto")

# NLTK downloads
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Custom CSS for Dark Mode
st.markdown("""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
@import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');

body, .stApp {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
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
    transition: all 0.2s ease-in-out !important;
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
    transition: all 0.2s ease-in-out !important;
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
            # Users table
            db.c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'teacher', 'student')),
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    is_approved BOOLEAN NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Assignments table with deadline
            db.c.execute('''
                CREATE TABLE IF NOT EXISTS assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    teacher_id INTEGER,
                    subject TEXT NOT NULL,
                    question TEXT NOT NULL,
                    model_answer TEXT NOT NULL,
                    keywords TEXT,
                    deadline TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(teacher_id) REFERENCES users(id) ON DELETE CASCADE
                )
            ''')

            # Check if deadline column exists, add if not
            db.c.execute("PRAGMA table_info(assignments)")
            columns = [col[1] for col in db.c.fetchall()]
            if 'deadline' not in columns:
                db.c.execute('ALTER TABLE assignments ADD COLUMN deadline TIMESTAMP')

            # Submissions table
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

            # Initialize admin user
            admin_username = 'Akshith'
            admin_password = 'admin123'
            db.c.execute('SELECT id FROM users WHERE username = ?', (admin_username,))
            if not db.c.fetchone():
                hashed_password = bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt()).decode()
                db.c.execute('INSERT INTO users (username, password, role, is_active, is_approved) VALUES (?, ?, ?, ?, ?)',
                             (admin_username, hashed_password, 'admin', 1, 1))
        except sqlite3.Error as e:
            st.error(f"Database initialization error: {e}")
            logger.error(f"Database initialization error: {e}")

# Initialize database
init_db()

# Password hashing
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Password verification
def verify_password(password, hashed):
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

# User authentication
def authenticate(username, password):
    with Database() as db:
        try:
            db.c.execute('SELECT id, username, password, role, is_approved FROM users WHERE username = ? AND is_active = 1', (username,))
            user = db.c.fetchone()
            if user and verify_password(password, user[2]):
                if user[3] != 'admin' and not user[4]:
                    return None, "Account not approved by admin. Please wait or contact the administrator."
                return {'id': user[0], 'username': user[1], 'role': user[3]}, None
        except sqlite3.Error as e:
            logger.error(f"Authentication error: {e}")
            return None, f"Database error: {e}"
    return None, "Invalid username or password."

# Check if user is rejected
def is_rejected(username):
    with Database() as db:
        try:
            db.c.execute('SELECT is_approved FROM users WHERE username = ? AND is_active = 1', (username,))
            result = db.c.fetchone()
            return result and result[0] == 0
        except sqlite3.Error as e:
            logger.error(f"Rejection check error: {e}")
            return False

# Clean up orphaned records
def cleanup_orphaned_records():
    with Database() as db:
        try:
            db.c.execute('DELETE FROM submissions WHERE student_id NOT IN (SELECT id FROM users)')
            db.c.execute('DELETE FROM assignments WHERE teacher_id NOT IN (SELECT id FROM users)')
        except sqlite3.Error as e:
            st.error(f"Database cleanup error: {e}")
            logger.error(f"Database cleanup error: {e}")

# Model loading
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_language_tool():
    return language_tool_python.LanguageTool('en-US')

# Preprocessing and evaluation functions
def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

@st.cache_data
def encode_answer(_model, text):
    return _model.encode(text, convert_to_tensor=True)

def semantic_similarity(ans1, ans2, confidence_threshold=0.5):
    if not ans1 or not ans2:
        return 0.0, 0.0
    model = load_model()
    embedding1 = encode_answer(model, ans1)
    embedding2 = encode_answer(model, ans2)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    confidence = round(float(similarity[0][0]) * 100, 2)
    adjusted_confidence = max(0, confidence - (100 * confidence_threshold))
    return confidence, adjusted_confidence

def score_similarity(similarity):
    if similarity < 10:
        return 0, "Irrelevant answer"
    elif similarity >= 85:
        return 10, "Excellent answer"
    elif similarity >= 70:
        return 8, "Good understanding"
    elif similarity >= 50:
        return 6, "Fair attempt"
    else:
        return 4, "Needs improvement"

def grammar_score(text):
    if not isinstance(text, str) or not text.strip():
        return 0, "Empty answer", []
    tool = load_language_tool()
    words = len(text.split())
    matches = tool.check(text)
    errors = len(matches)
    error_rate = min(errors / words, 0.5) if words > 0 else 0.5
    if errors == 0:
        return 10, "Perfect grammar", matches
    elif error_rate < 0.02:
        return 10, "Excellent grammar", matches
    elif error_rate < 0.05:
        return 8, "Minor grammatical issues", matches
    elif error_rate < 0.1:
        return 6, "Noticeable errors", matches
    else:
        return 4, "Poor grammar", matches

def length_penalty(text):
    if not isinstance(text, str) or not text.strip():
        return 0.8, "Empty answer"
    words = len(text.split())
    if words < 3:
        return 0.9, "Answer very short"
    elif words > 200:
        return 0.9, "Answer too long"
    return 1.0, "Appropriate length"

def get_synonyms(word):
    synonyms = {word}
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def keyword_match(student, keywords, subject):
    if not isinstance(student, str) or not student.strip():
        return 0, "Empty answer"
    student_words = set(preprocess(student).split())
    matched = []
    for kw in keywords:
        kw_parts = kw.split()
        kw_synonyms = set()
        for part in kw_parts:
            kw_synonyms.update(get_synonyms(part))
        if any(syn in student_words for syn in kw_synonyms):
            matched.append(kw)
    match_rate = len(matched) / len(keywords) if keywords else 0
    if match_rate >= 0.9:
        return 10, "All keywords matched"
    elif match_rate >= 0.7:
        return 8, "Most keywords matched"
    elif match_rate >= 0.5:
        return 6, "Some keywords matched"
    else:
        return 4, "Important concepts missing"

def check_plagiarism(submissions, threshold=95.0):
    model = load_model()
    plagiarism_results = []
    for i, sub1 in enumerate(submissions):
        for j, sub2 in enumerate(submissions[i+1:], start=i+1):
            if sub1[0] != sub2[0]:  # Different submission IDs
                sim, _ = semantic_similarity(sub1[1], sub2[1])
                if sim > threshold:
                    plagiarism_results.append({
                        'submission_id_1': sub1[0],
                        'submission_id_2': sub2[0],
                        'similarity': sim,
                        'student_1': sub1[2],
                        'student_2': sub2[3]
                    })
    return plagiarism_results

def validate_keywords(keywords):
    allowed_chars = re.compile(r'^[a-zA-Z0-9\s²³=+\-*/()[\]^]+$')
    for kw in keywords:
        if not (allowed_chars.match(kw) and len(kw) <= 50):
            return False, f"Invalid keyword: '{kw}' (must be alphanumeric, spaces, or math symbols [²³=+-*/()[]^], ≤50 chars)"
    return True, ""

def validate_subject(subject):
    allowed_subjects = {
        'biology': 'Biology', 'bio': 'Biology',
        'physics': 'Physics', 'phys': 'Physics',
        'history': 'History', 'hist': 'History',
        'literature': 'Literature', 'lit': 'Literature',
        'mathematics': 'Mathematics', 'math': 'Mathematics', 'maths': 'Mathematics',
        'chemistry': 'Chemistry', 'chem': 'Chemistry',
        'geometry': 'Geometry', 'geom': 'Geometry',
        'other': 'Other'
    }
    return allowed_subjects.get(str(subject).lower().strip(), None)

def validate_question(question):
    if not isinstance(question, str) or not question.strip():
        return False, "Question cannot be empty."
    if len(question) > 1000:
        return False, "Question must be 1000 characters or less."
    return True, ""

def validate_deadline(deadline):
    try:
        # Try parsing DD/MM/YYYY and DD-MM-YYYY
        for fmt in ['%d/%m/%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S']:
            try:
                deadline_dt = dt.datetime.strptime(deadline + " 23:59:59", fmt)
                if deadline_dt < dt.datetime.now():
                    return False, "Deadline must be in the future."
                # Convert to YYYY-MM-DD HH:MM:SS for database storage
                return True, deadline_dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
        return False, "Invalid deadline format. Use DD/MM/YYYY or DD-MM-YYYY (e.g., 31/12/2025 or 31-12-2025)."
    except Exception as e:
        return False, f"Error parsing deadline: {e}"

def evaluate_answer(model_answer, student_answer, keywords, subject):
    if not model_answer or not isinstance(model_answer, str) or not model_answer.strip():
        return {
            'Similarity (%)': 0.0, 'Confidence (%)': 0.0, 'Similarity Score': 0, 'Similarity Feedback': 'No model answer provided',
            'Grammar Score': 0, 'Grammar Feedback': 'No answer evaluated', 'Keyword Score': 0, 'Keyword Feedback': 'No keywords evaluated',
            'Matched Keywords': 'None', 'Length Penalty': 0.8, 'Length Feedback': 'No answer evaluated', 'Final Score': 0.0, 'Grammar Matches': [],
            'Suggestions': 'Please provide a model answer.'
        }
    if not student_answer or not isinstance(student_answer, str) or not student_answer.strip():
        return {
            'Similarity (%)': 0.0, 'Confidence (%)': 0.0, 'Similarity Score': 0, 'Similarity Feedback': 'Empty answer',
            'Grammar Score': 0, 'Grammar Feedback': 'Empty answer', 'Keyword Score': 0, 'Keyword Feedback': 'Empty answer',
            'Matched Keywords': 'None', 'Length Penalty': 0.8, 'Length Feedback': 'Empty answer', 'Final Score': 0.0, 'Grammar Matches': [],
            'Suggestions': 'Please provide a response.'
        }

    # Calculate initial scores
    similarity, confidence = semantic_similarity(student_answer, model_answer)
    sim_score, sim_feedback = score_similarity(similarity)
    grammar_points, grammar_feedback, grammar_matches = grammar_score(student_answer)
    keyword_points, keyword_feedback = keyword_match(student_answer, keywords, subject)
    length_factor, length_feedback = length_penalty(student_answer)
    matched_keywords = [kw for kw in keywords if any(syn in set(preprocess(student_answer).split()) for syn in get_synonyms(kw.split()[0]))]

    # Topic relevance check
    is_relevant = False
    if similarity >= 10 and keyword_points >= 4:
        is_relevant = True

    # Adjust scores based on relevance and correctness
    if not is_relevant:
        sim_score = 0
        sim_feedback = "Unrelated to the topic"
        keyword_points = 0
        keyword_feedback = "No relevant keywords found"
        final_score = 0
        suggestions = ["The answer is unrelated to the topic. Please review the question and provide a relevant response."]
    else:
        if sim_score < 50 and keyword_points < 6:
            sim_score = 2
            sim_feedback = "Incorrect but topic-related"
            keyword_points = 2
            keyword_feedback = "Some topic relevance but incorrect"
            final_score = 2
            suggestions = ["The answer is incorrect. Review the model answer and key concepts."]
        else:
            sim_weight, grammar_weight, keyword_weight = 0.4, 0.3, 0.3
            suggestions = []
            if sim_score < 6:
                suggestions.append("Review the core concepts and ensure your answer aligns with the expected response.")
            if grammar_points < 6:
                suggestions.append("Check grammar and sentence structure for clarity.")
            if keyword_points < 6:
                suggestions.append("Include key concepts or synonyms in your answer.")
            if length_factor < 1.0:
                suggestions.append("Adjust answer length to be concise yet complete.")
            total_weight = sim_weight + grammar_weight + keyword_weight
            if sim_score == 0:
                final_score = 0
            else:
                raw_score = (sim_score * sim_weight + grammar_points * grammar_weight + keyword_points * keyword_weight) / total_weight
                final_score = min(10.0, max(0.0, round(raw_score * length_factor, 1)))

    return {
        'Similarity (%)': similarity, 'Confidence (%)': confidence, 'Similarity Score': sim_score, 'Similarity Feedback': sim_feedback,
        'Grammar Score': grammar_points, 'Grammar Feedback': grammar_feedback, 'Keyword Score': keyword_points, 'Keyword Feedback': keyword_feedback,
        'Matched Keywords': ', '.join(matched_keywords) or 'None', 'Length Penalty': length_factor, 'Length Feedback': length_feedback,
        'Final Score': final_score, 'Grammar Matches': [{'ruleId': m.ruleId, 'message': m.message, 'context': m.context} for m in grammar_matches],
        'Suggestions': ' '.join(suggestions)
    }

# Registration function
def register_user(username, password, role):
    if len(username) < 4 or len(password) < 6:
        return False, "Username ≥4 chars, password ≥6 chars required."
    if role not in ['teacher', 'student']:
        return False, "Invalid role."
    with Database() as db:
        try:
            db.c.execute('INSERT INTO users (username, password, role, is_active, is_approved) VALUES (?, ?, ?, ?, ?)',
                         (username, hash_password(password), role.lower(), 1, 0))
            return True, "Registration successful! Please wait for admin approval."
        except sqlite3.IntegrityError:
            return False, "Username already exists."
        except sqlite3.Error as e:
            return False, f"Database error: {e}"

# Admin user management
def manage_users():
    st.subheader("Pending Approvals")
    with Database() as db:
        try:
            db.c.execute('SELECT id, username, role, created_at FROM users WHERE is_approved = 0 AND is_active = 1')
            pending = db.c.fetchall()
            if pending:
                pending_df = pd.DataFrame(pending, columns=['ID', 'Username', 'Role', 'Registered On'])
                st.dataframe(pending_df, use_container_width=True)
                user_id = st.number_input("User ID to Approve/Reject", min_value=1, step=1, key="approve_user_id_admin")
                action = st.selectbox("Action", ["Approve", "Reject"], key="approve_action_admin")
                if st.button("Submit Action", key="submit_action_admin"):
                    if action == "Approve":
                        db.c.execute('UPDATE users SET is_approved = 1 WHERE id = ?', (user_id,))
                        st.success("User approved!")
                    else:
                        db.c.execute('UPDATE users SET is_approved = 0, is_active = 0 WHERE id = ?', (user_id,))
                        st.success("User rejected!")
                    st.rerun()
            else:
                st.info("No pending approvals.")
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")

    st.subheader("All Users")
    with Database() as db:
        try:
            db.c.execute('SELECT id, username, role, is_active, is_approved FROM users WHERE role != "admin"')
            users = db.c.fetchall()
            if users:
                users_df = pd.DataFrame(users, columns=['ID', 'Username', 'Role', 'Active', 'Approved'])
                st.dataframe(users_df, use_container_width=True)
            else:
                st.info("No users registered.")
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")

# Get assignments with formatted deadline
def get_assignments(user_id, role):
    with Database() as db:
        try:
            # Check if deadline column exists
            db.c.execute("PRAGMA table_info(assignments)")
            columns = [col[1] for col in db.c.fetchall()]
            select_deadline = "a.deadline" if 'deadline' in columns else "NULL as deadline"
            
            if role == 'student':
                db.c.execute(f'''
                    SELECT a.id, a.subject, a.question, a.created_at, t.username, 
                           (SELECT COUNT(*) FROM submissions s WHERE s.assignment_id = a.id AND s.student_id = ?), 
                           {select_deadline}
                    FROM assignments a
                    JOIN users t ON a.teacher_id = t.id
                ''', (user_id,))
            else:
                db.c.execute(f'''
                    SELECT a.id, a.subject, a.question, a.created_at, t.username, 
                           (SELECT COUNT(*) FROM submissions s WHERE s.assignment_id = a.id), 
                           {select_deadline}
                    FROM assignments a
                    JOIN users t ON a.teacher_id = t.id
                ''')
            assignments = db.c.fetchall()
            # Convert deadline to DD/MM/YYYY
            formatted_assignments = []
            for assignment in assignments:
                deadline = assignment[6]
                if deadline:
                    try:
                        deadline_dt = dt.datetime.strptime(deadline, '%Y-%m-%d %H:%M:%S')
                        formatted_deadline = deadline_dt.strftime('%d/%m/%Y')
                    except ValueError:
                        formatted_deadline = deadline
                else:
                    formatted_deadline = None
                formatted_assignments.append(assignment[:6] + (formatted_deadline,))
            return formatted_assignments
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            logger.error(f"Database error in get_assignments: {e}")
            return []

# Get submissions with student_id and assignment_id
def get_submissions(user_id, role):
    with Database() as db:
        try:
            if role == 'student':
                db.c.execute('''
                    SELECT s.id, a.subject, a.question, s.answer, s.submitted_at, s.status, s.evaluation_result, u.username, a.deadline, s.student_id, s.assignment_id
                    FROM submissions s
                    JOIN assignments a ON s.assignment_id = a.id
                    JOIN users u ON a.teacher_id = u.id
                    WHERE s.student_id = ?
                ''', (user_id,))
            elif role == 'teacher':
                db.c.execute('''
                    SELECT s.id, a.subject, a.question, s.answer, s.submitted_at, s.status, s.evaluation_result, u.username, a.deadline, s.student_id, s.assignment_id
                    FROM submissions s
                    JOIN assignments a ON s.assignment_id = a.id
                    JOIN users u ON s.student_id = u.id
                    WHERE a.teacher_id = ?
                ''', (user_id,))
            else:  # admin
                db.c.execute('''
                    SELECT s.id, a.subject, a.question, s.answer, s.submitted_at, s.status, s.evaluation_result, u.username, a.deadline, s.student_id, s.assignment_id
                    FROM submissions s
                    JOIN assignments a ON s.assignment_id = a.id
                    JOIN users u ON s.student_id = u.id
                ''')
            submissions = db.c.fetchall()
            # Format deadline in submissions
            formatted_submissions = []
            for sub in submissions:
                deadline = sub[8]
                if deadline:
                    try:
                        deadline_dt = dt.datetime.strptime(deadline, '%Y-%m-%d %H:%M:%S')
                        formatted_deadline = deadline_dt.strftime('%d/%m/%Y')
                    except ValueError:
                        formatted_deadline = deadline
                else:
                    formatted_deadline = None
                formatted_submissions.append(sub[:8] + (formatted_deadline,) + sub[9:])
            return formatted_submissions
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            logger.error(f"Database error in get_submissions: {e}")
            return []

# Check if student has already submitted
def has_submitted(student_id, assignment_id):
    with Database() as db:
        try:
            db.c.execute('SELECT COUNT(*) FROM submissions WHERE student_id = ? AND assignment_id = ?', (student_id, assignment_id))
            return db.c.fetchone()[0] > 0
        except sqlite3.Error as e:
            logger.error(f"Submission check error: {e}")
            return False

# Check if assignment deadline has passed
def is_deadline_passed(assignment_id):
    with Database() as db:
        try:
            # Check if deadline column exists
            db.c.execute("PRAGMA table_info(assignments)")
            columns = [col[1] for col in db.c.fetchall()]
            if 'deadline' not in columns:
                return False  # No deadline column, so no deadline restrictions
            db.c.execute('SELECT deadline FROM assignments WHERE id = ?', (assignment_id,))
            deadline = db.c.fetchone()
            if deadline and deadline[0]:
                deadline_dt = dt.datetime.strptime(deadline[0], '%Y-%m-%d %H:%M:%S')
                return deadline_dt < dt.datetime.now()
            return False
        except (sqlite3.Error, ValueError) as e:
            logger.error(f"Deadline check error: {e}")
            return False

# Bulk assignment creation from CSV
def create_assignments_from_csv(file, teacher_id):
    try:
        df = pd.read_csv(file)
        required_columns = ['subject', 'question', 'model_answer', 'keywords']
        if not all(col in df.columns for col in required_columns):
            return False, "CSV must contain columns: subject, question, model_answer, keywords"
        successes = 0
        errors = []
        with Database() as db:
            for _, row in df.iterrows():
                subject = validate_subject(row['subject'])
                if not subject:
                    errors.append(f"Invalid subject: {row['subject']}")
                    continue
                is_valid_question, question_error = validate_question(row['question'])
                if not is_valid_question:
                    errors.append(question_error)
                    continue
                keywords = [kw.strip() for kw in row['keywords'].split(",") if kw.strip()] if row['keywords'] else []
                is_valid_keywords, keyword_error = validate_keywords(keywords)
                if not is_valid_keywords:
                    errors.append(keyword_error)
                    continue
                deadline = row.get('deadline', None)
                if deadline:
                    is_valid_deadline, deadline_flag = validate_deadline(deadline)
                    if not is_valid_deadline:
                        errors.append(deadline_flag)
                        continue
                    deadline = deadline_flag  # Use the formatted timestamp
                try:
                    db.c.execute('INSERT INTO assignments (teacher_id, subject, question, model_answer, keywords, deadline) VALUES (?, ?, ?, ?, ?, ?)',
                                 (teacher_id, subject, row['question'], row['model_answer'], ",".join(keywords), deadline))
                    successes += 1
                except sqlite3.Error as e:
                    errors.append(f"Error inserting assignment: {e}")
        return successes > 0, f"Created {successes} assignments. Errors: {', '.join(errors) if errors else 'None'}"
    except Exception as e:
        return False, f"Error processing CSV: {e}"

# Student analytics
def get_student_analytics(student_id):
    with Database() as db:
        try:
            db.c.execute('''
                SELECT s.evaluation_result
                FROM submissions s
                WHERE s.student_id = ? AND s.evaluation_result IS NOT NULL
            ''', (student_id,))
            submissions = db.c.fetchall()
            if not submissions:
                return None
            scores = []
            grammar_scores = []
            similarity_scores = []
            for sub in submissions:
                try:
                    result = json.loads(sub[0])
                    scores.append(result.get('Final Score', 0.0))
                    grammar_scores.append(result.get('Grammar Score', 0))
                    similarity_scores.append(result.get('Similarity Score', 0))
                except json.JSONDecodeError:
                    continue
            if not scores:
                return None
            avg_score = sum(scores) / len(scores)
            grammar_trend = "Improving" if len(grammar_scores) > 1 and grammar_scores[-1] > grammar_scores[0] else "Stable or Declining"
            similarity_trend = "Improving" if len(similarity_scores) > 1 and similarity_scores[-1] > similarity_scores[0] else "Stable or Declining"
            return {
                'Average Score': round(avg_score, 1),
                'Grammar Trend': grammar_trend,
                'Similarity Trend': similarity_trend,
                'Total Submissions': len(scores)
            }
        except sqlite3.Error as e:
            st.error(f"Analytics error: {e}")
            return None

# View submission details
def view_details(submission_id):
    st.session_state.selected_submission_id = submission_id
    st.rerun()

# Main app
if 'user' not in st.session_state:
    st.session_state.user = None
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'last_registered_username' not in st.session_state:
    st.session_state.last_registered_username = None
if 'selected_submission_id' not in st.session_state:
    st.session_state.selected_submission_id = None

st.title("📘 Subjective Answer Evaluator")

if not st.session_state.user:
    st.header("🔐 Authentication")
    auth_action = st.radio("Select Action", ["Login", "Register"], key="auth_action_radio")
    if auth_action != st.session_state.get('auth_action', "Login"):
        st.session_state.auth_action = auth_action
        st.rerun()

    with st.form("auth_form"):
        # Prepopulate username if last registered
        username = st.text_input("Username", value=st.session_state.last_registered_username if st.session_state.last_registered_username else "", key="login_username_auth")
        password = st.text_input("Password", type="password", key="login_password_auth")
        if auth_action == "Register":
            role = st.selectbox("Role", ["Teacher", "Student"], key="register_role_auth")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
            elif auth_action == "Login":
                user, error = authenticate(username, password)
                if user:
                    st.session_state.user = user
                    st.success(f"Welcome, {username}!")
                    cleanup_orphaned_records()
                    st.rerun()
                else:
                    st.error(error)
            else:
                success, message = register_user(username, password, role.lower())
                if success:
                    st.success(message)
                    st.session_state.last_registered_username = username  # Store the registered username
                    st.session_state.auth_action = "Login"  # Switch back to login
                    st.rerun()
                else:
                    st.error(message)
else:
    user = st.session_state.user
    st.sidebar.title(f"👤 {user['username']} ({user['role'].capitalize()})")
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.user = None
        st.session_state.notifications = []
        st.session_state.auth_action = "Login"
        st.session_state.last_registered_username = None
        st.session_state.selected_submission_id = None
        st.rerun()

    # Notifications
    if st.session_state.notifications:
        with st.sidebar.expander("🔔 Notifications", expanded=True):
            for notif in st.session_state.notifications:
                st.write(notif)
        if st.sidebar.button("Clear Notifications", key="clear_notifications"):
            st.session_state.notifications = []
            st.rerun()

    if user['role'] == 'admin':
        st.header("📘 Admin Dashboard")
        manage_users()

        st.subheader("All Assignments")
        assignments = get_assignments(user['id'], user['role'])
        if assignments:
            assignments_df = pd.DataFrame(assignments, columns=['ID', 'Subject', 'Question', 'Created At', 'Teacher', 'Submissions', 'Deadline'])
            st.dataframe(assignments_df, use_container_width=True)
            # Add CSV download for assignments
            csv_buffer = io.StringIO()
            assignments_df.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_NONNUMERIC, date_format='%d/%m/%Y')
            st.download_button(
                label="📑 Download Assignments (CSV)",
                data=csv_buffer.getvalue(),
                file_name="assignments.csv",
                mime="text/csv",
                key="download_assignments_admin"
            )
        else:
            st.info("No assignments created yet.")

        with st.expander("Evaluate Submissions", expanded=True):
            submissions = get_submissions(user['id'], user['role'])
            pending_submissions = [(s[0], s[1], s[2], s[3], s[4], s[7], s[9], s[10]) for s in submissions if s[5] == 'Pending']
            if pending_submissions:
                submissions_df = pd.DataFrame(pending_submissions, columns=['ID', 'Subject', 'Question', 'Answer', 'Submitted At', 'Student', 'Student ID', 'Assignment ID'])
                st.dataframe(submissions_df, use_container_width=True)
                submission_id = st.number_input("Submission ID to Evaluate", min_value=1, step=1, key="admin_eval_submission_id")
                if submission_id in [s[0] for s in pending_submissions]:
                    with Database() as db:
                        db.c.execute('''
                            SELECT s.answer, a.question, a.model_answer, a.keywords, a.subject
                            FROM submissions s
                            JOIN assignments a ON s.assignment_id = a.id
                            WHERE s.id = ? AND s.status = 'Pending'
                        ''', (submission_id,))
                        submission = db.c.fetchone()
                        if submission:
                            student_answer, question, model_answer, keywords_str, subject = submission
                            keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()] if keywords_str else []
                            st.text_area("❓ Question", value=question, height=100, disabled=True, key=f"question_admin_{submission_id}")
                            st.text_area("🧑‍🎓 Student's Answer", value=student_answer, height=150, disabled=True, key=f"student_answer_admin_{submission_id}")
                            st.text_area("✍️ Model Answer", value=model_answer, height=150, disabled=True, key=f"model_answer_admin_{submission_id}")
                            st.text_input("🔑 Keywords", value=", ".join(keywords), disabled=True, key=f"keywords_admin_{submission_id}")
                            if st.button("📊 Evaluate Answer", type="primary", key=f"eval_button_admin_{submission_id}"):
                                with st.spinner("Evaluating answer..."):
                                    result = evaluate_answer(model_answer, student_answer, keywords, subject)
                                    try:
                                        evaluation_result = json.dumps(result)
                                        db.c.execute('UPDATE submissions SET status = ?, evaluation_result = ? WHERE id = ?',
                                                     ("Evaluated", evaluation_result, submission_id))
                                        st.success("✅ Evaluation Complete")
                                        st.session_state.notifications.append(f"Submission {submission_id} evaluated at {datetime.now().strftime('%H:%M:%S')}")
                                        st.markdown("### 📈 Evaluation Summary")
                                        with st.container():
                                            st.write(f"<i class='fas fa-brain'></i> <b>Semantic Similarity:</b> {result['Similarity (%)']:.1f}% → Score: {result['Similarity Score']}/10 → <i>{result['Similarity Feedback']}</i>", unsafe_allow_html=True)
                                            st.write(f"<i class='fas fa-pencil-alt'></i> <b>Grammar Score:</b> {result['Grammar Score']}/10 → <i>{result['Grammar Feedback']}</i>", unsafe_allow_html=True)
                                            if result['Grammar Matches']:
                                                st.markdown("#### <i class='fas fa-exclamation-circle'></i> Grammar Issues (Top 5)", unsafe_allow_html=True)
                                                for match in result['Grammar Matches'][:5]:
                                                    st.write(f"- {match['ruleId']}: {match['message']} (at: {match['context']})")
                                            st.write(f"<i class='fas fa-key'></i> <b>Keyword Match Score:</b> {result['Keyword Score']}/10 → <i>{result['Keyword Feedback']}</i>", unsafe_allow_html=True)
                                            st.write(f"<i class='fas fa-tags'></i> <b>Matched Keywords:</b> {result['Matched Keywords']}", unsafe_allow_html=True)
                                            st.write(f"<i class='fas fa-ruler'></i> <b>Length Penalty:</b> <i>{result['Length Feedback']}</i>", unsafe_allow_html=True)
                                            st.write(f"<i class='fas fa-star'></i> <b>Final Score:</b> {result['Final Score']}/10", unsafe_allow_html=True)
                                            if result['Suggestions']:
                                                st.markdown("#### <i class='fas fa-lightbulb'></i> Suggestions", unsafe_allow_html=True)
                                                st.write(result['Suggestions'])
                                    except json.JSONDecodeError as e:
                                        st.error(f"Error serializing evaluation result: {e}")
                        else:
                            st.error("Submission not found or already evaluated.")
                else:
                    st.error("Please select a valid pending submission ID from the list above.")
            else:
                st.info("No pending submissions to evaluate.")

    elif user['role'] == 'teacher':
        st.header("📘 Teacher Dashboard")
        st.markdown("Create assignments and evaluate student submissions.")
        with st.expander("📝 Create Assignment", expanded=True):
            subject = st.selectbox("📚 Subject", ["Biology", "Physics", "History", "Literature", "Mathematics", "Chemistry", "Geometry", "Other"], key="teacher_subject_create")
            question = st.text_area("❓ Question (max 1000 chars)", height=100, max_chars=1000, placeholder="Enter the question...", key="teacher_question_create")
            model_answer = st.text_area("✍️ Model Answer (max 1000 chars)", height=150, max_chars=1000, placeholder="Enter the ideal answer...", key="teacher_model_answer_create")
            keywords_input = st.text_input("🔑 Keywords (comma-separated)", placeholder="e.g., carbon dioxide, chlorophyll", key="teacher_keywords_create")
            deadline = st.text_input("📅 Deadline (DD/MM/YYYY or DD-MM-YYYY)", placeholder="e.g., 31/12/2025 or 31-12-2025", key="teacher_deadline_create")
            if st.button("📤 Create Assignment", key="create_assignment_button"):
                if not question or not model_answer:
                    st.error("Please provide both a question and a model answer.")
                else:
                    is_valid_question, question_error = validate_question(question)
                    keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()] if keywords_input else []
                    is_valid_keywords, keyword_error = validate_keywords(keywords)
                    is_valid_deadline, deadline_flag = validate_deadline(deadline) if deadline else (True, None)
                    if not is_valid_question:
                        st.error(question_error)
                    elif not is_valid_keywords:
                        st.error(keyword_error)
                    elif not is_valid_deadline:
                        st.error(deadline_flag)
                    else:
                        with Database() as db:
                            try:
                                db.c.execute('INSERT INTO assignments (teacher_id, subject, question, model_answer, keywords, deadline) VALUES (?, ?, ?, ?, ?, ?)',
                                             (user['id'], subject, question, model_answer, ",".join(keywords), deadline_flag))
                                st.success("Assignment created successfully!")
                                st.session_state.notifications.append(f"New assignment in {subject} created at {datetime.now().strftime('%H:%M:%S')}")
                                st.rerun()
                            except sqlite3.Error as e:
                                st.error(f"Error creating assignment: {e}")

            # CSV Upload for Bulk Assignment Creation
            st.subheader("📂 Bulk Assignment Creation")
            st.markdown("Upload a CSV with columns: subject, question, model_answer, keywords, deadline (DD/MM/YYYY or DD-MM-YYYY).")
            csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload_teacher")
            if csv_file and st.button("Upload and Create Assignments", key="upload_csv_button"):
                success, message = create_assignments_from_csv(csv_file, user['id'])
                if success:
                    st.success(message)
                    st.session_state.notifications.append(f"Bulk assignment creation completed at {datetime.now().strftime('%H:%M:%S')}")
                else:
                    st.error(message)

        with st.expander("📊 Evaluate Submissions", expanded=True):
            submissions = get_submissions(user['id'], user['role'])
            pending_submissions = [(s[0], s[1], s[2], s[3], s[4], s[7], s[9], s[10]) for s in submissions if s[5] == 'Pending']
            if pending_submissions:
                submissions_df = pd.DataFrame(pending_submissions, columns=['Submission ID', 'Subject', 'Question', 'Answer', 'Submitted At', 'Student', 'Student ID', 'Assignment ID'])
                st.dataframe(submissions_df, use_container_width=True)
                # Individual Evaluation
                submission_id = st.number_input("Submission ID to Evaluate", min_value=1, step=1, key="teacher_eval_submission_id")
                if submission_id in [s[0] for s in pending_submissions]:
                    with Database() as db:
                        db.c.execute('''
                            SELECT s.answer, a.question, a.model_answer, a.keywords, a.subject
                            FROM submissions s
                            JOIN assignments a ON s.assignment_id = a.id
                            WHERE s.id = ? AND s.status = 'Pending'
                        ''', (submission_id,))
                        submission = db.c.fetchone()
                        if submission:
                            student_answer, question, model_answer, keywords_str, subject = submission
                            keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()] if keywords_str else []
                            st.text_area("❓ Question", value=question, height=100, disabled=True, key=f"question_teacher_eval_{submission_id}")
                            st.text_area("🧑‍🎓 Student's Answer", value=student_answer, height=150, disabled=True, key=f"student_answer_teacher_eval_{submission_id}")
                            st.text_area("✍️ Model Answer", value=model_answer, height=150, disabled=True, key=f"model_answer_teacher_eval_{submission_id}")
                            st.text_input("🔑 Keywords", value=", ".join(keywords), disabled=True, key=f"keywords_teacher_eval_{submission_id}")
                            if st.button("📊 Evaluate Answer", type="primary", key=f"eval_button_teacher_{submission_id}"):
                                with st.spinner("Evaluating answer..."):
                                    result = evaluate_answer(model_answer, student_answer, keywords, subject)
                                    try:
                                        evaluation_result = json.dumps(result)
                                        db.c.execute('UPDATE submissions SET status = ?, evaluation_result = ? WHERE id = ?',
                                                     ("Evaluated", evaluation_result, submission_id))
                                        st.success("✅ Evaluation Complete")
                                        st.session_state.notifications.append(f"Submission {submission_id} evaluated at {datetime.now().strftime('%H:%M:%S')}")
                                        st.markdown("### 📈 Evaluation Summary")
                                        with st.container():
                                            st.write(f"<i class='fas fa-brain'></i> <b>Semantic Similarity:</b> {result['Similarity (%)']:.1f}% → Score: {result['Similarity Score']}/10 → <i>{result['Similarity Feedback']}</i>", unsafe_allow_html=True)
                                            st.write(f"<i class='fas fa-pencil-alt'></i> <b>Grammar Score:</b> {result['Grammar Score']}/10 → <i>{result['Grammar Feedback']}</i>", unsafe_allow_html=True)
                                            if result['Grammar Matches']:
                                                st.markdown("#### <i class='fas fa-exclamation-circle'></i> Grammar Issues (Top 5)", unsafe_allow_html=True)
                                                for match in result['Grammar Matches'][:5]:
                                                    st.write(f"- {match['ruleId']}: {match['message']} (at: {match['context']})")
                                            st.write(f"<i class='fas fa-key'></i> <b>Keyword Match Score:</b> {result['Keyword Score']}/10 → <i>{result['Keyword Feedback']}</i>", unsafe_allow_html=True)
                                            st.write(f"<i class='fas fa-tags'></i> <b>Matched Keywords:</b> {result['Matched Keywords']}", unsafe_allow_html=True)
                                            st.write(f"<i class='fas fa-ruler'></i> <b>Length Penalty:</b> <i>{result['Length Feedback']}</i>", unsafe_allow_html=True)
                                            st.write(f"<i class='fas fa-star'></i> <b>Final Score:</b> {result['Final Score']}/10", unsafe_allow_html=True)
                                            if result['Suggestions']:
                                                st.markdown("#### <i class='fas fa-lightbulb'></i> Suggestions", unsafe_allow_html=True)
                                                st.write(result['Suggestions'])
                                    except json.JSONDecodeError as e:
                                        st.error(f"Error serializing evaluation result: {e}")
                        else:
                            st.error("Submission not found or already evaluated.")
                else:
                    st.error("Please select a valid pending submission ID from the list above.")
                # Batch Evaluation by Assignment
                st.subheader("📂 Batch Evaluation by Assignment")
                assignment_id = st.number_input("Assignment ID to Evaluate All Pending Submissions", min_value=1, step=1, key="batch_eval_assignment_id_teacher")
                if st.button("Evaluate All Pending Submissions for Assignment", key="batch_eval_button_teacher"):
                    with Database() as db:
                        db.c.execute('''
                            SELECT s.id, s.answer, a.model_answer, a.keywords, a.subject, u.username, s.student_id
                            FROM submissions s
                            JOIN assignments a ON s.assignment_id = a.id
                            JOIN users u ON s.student_id = u.id
                            WHERE s.assignment_id = ? AND s.status = 'Pending' AND a.teacher_id = ?
                        ''', (assignment_id, user['id']))
                        pending_subs = db.c.fetchall()
                        if pending_subs:
                            # Plagiarism check
                            plagiarism_results = check_plagiarism([(sub[0], sub[1], sub[5], sub[6]) for sub in pending_subs])
                            if plagiarism_results:
                                st.warning("⚠️ Potential plagiarism detected:")
                                for pr in plagiarism_results:
                                    st.write(f"Submissions {pr['submission_id_1']} (Student ID: {pr['student_1']}) and {pr['submission_id_2']} (Student ID: {pr['student_2']}) are {pr['similarity']:.1f}% similar.")
                            for sub in pending_subs:
                                submission_id, student_answer, model_answer, keywords_str, subject, _, _ = sub
                                keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()] if keywords_str else []
                                result = evaluate_answer(model_answer, student_answer, keywords, subject)
                                try:
                                    evaluation_result = json.dumps(result)
                                    db.c.execute('UPDATE submissions SET status = ?, evaluation_result = ? WHERE id = ?',
                                                 ("Evaluated", evaluation_result, submission_id))
                                except json.JSONDecodeError as e:
                                    st.error(f"Error serializing evaluation result for submission {submission_id}: {e}")
                            st.success(f"✅ Evaluated {len(pending_subs)} submissions for Assignment ID {assignment_id}")
                            st.session_state.notifications.append(f"Batch evaluation for Assignment {assignment_id} completed at {datetime.now().strftime('%H:%M:%S')}")
                            st.rerun()
                        else:
                            st.error("No pending submissions for this assignment.")
            else:
                st.info("No pending submissions to evaluate.")

    else:  # Student
        st.header("📘 Student Dashboard")
        st.markdown("View and submit assignments assigned by your teachers.")
        # Analytics Dashboard
        with st.expander("📊 Your Performance Analytics", expanded=True):
            analytics = get_student_analytics(user['id'])
            if analytics:
                st.write(f"**Average Score:** {analytics['Average Score']}/10")
                st.write(f"**Grammar Trend:** {analytics['Grammar Trend']}")
                st.write(f"**Similarity Trend:** {analytics['Similarity Trend']}")
                st.write(f"**Total Submissions:** {analytics['Total Submissions']}")
            else:
                st.info("No evaluated submissions yet.")
        assignments = get_assignments(user['id'], user['role'])
        pending_assignments = [(a[0], a[1], a[2], a[3], a[4], a[6]) for a in assignments if a[5] == 0]
        submitted_assignments = [(a[0], a[1], a[2], a[3], a[4], a[6]) for a in assignments if a[5] > 0]
        if pending_assignments:
            st.subheader("📋 Pending Assignments")
            pending_df = pd.DataFrame(pending_assignments, columns=['ID', 'Subject', 'Question', 'Created At', 'Teacher', 'Deadline'])
            st.dataframe(pending_df, use_container_width=True)
            # Add CSV download for pending assignments
            csv_buffer = io.StringIO()
            pending_df.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_NONNUMERIC, date_format='%d/%m/%Y')
            st.download_button(
                label="📑 Download Pending Assignments (CSV)",
                data=csv_buffer.getvalue(),
                file_name="pending_assignments.csv",
                mime="text/csv",
                key="download_pending_assignments_student"
            )
            assignment_id = st.number_input("Assignment ID to Submit", min_value=1, step=1, key="submit_assignment_id_student")
            if assignment_id:
                with Database() as db:
                    db.c.execute('SELECT question, subject, deadline FROM assignments WHERE id = ?', (assignment_id,))
                    assignment = db.c.fetchone()
                    if assignment:
                        question, subject, deadline = assignment
                        st.text_area("❓ Question", value=question, height=100, disabled=True, key=f"question_student_submit_{assignment_id}")
                        if deadline and is_deadline_passed(assignment_id):
                            st.error("The deadline for this assignment has passed.")
                        elif has_submitted(user['id'], assignment_id):
                            st.error("You have already submitted an answer for this assignment.")
                        else:
                            student_answer = st.text_area("🧑‍🎓 Your Answer (max 1000 chars)", height=150, max_chars=1000, placeholder="Enter your answer...", key=f"student_answer_submit_{assignment_id}")
                            if st.button("📤 Submit Answer", key=f"submit_answer_button_{assignment_id}"):
                                if not student_answer:
                                    st.error("Please provide an answer.")
                                else:
                                    try:
                                        db.c.execute('INSERT INTO submissions (student_id, assignment_id, answer, status) VALUES (?, ?, ?, ?)',
                                                     (user['id'], assignment_id, student_answer, "Pending"))
                                        st.success("Answer submitted successfully!")
                                        st.session_state.notifications.append(f"Assignment {assignment_id} submitted at {datetime.now().strftime('%H:%M:%S')}")
                                        st.rerun()
                                    except sqlite3.Error as e:
                                        st.error(f"Error submitting answer: {e}")
                    else:
                        st.error("Invalid assignment ID.")
        else:
            st.info("No pending assignments.")
        if submitted_assignments:
            st.subheader("✅ Submitted Assignments")
            submitted_df = pd.DataFrame(submitted_assignments, columns=['ID', 'Subject', 'Question', 'Created At', 'Teacher', 'Deadline'])
            st.dataframe(submitted_df, use_container_width=True)
            # Add CSV download for submitted assignments
            csv_buffer = io.StringIO()
            submitted_df.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_NONNUMERIC, date_format='%d/%m/%Y')
            st.download_button(
                label="📑 Download Submitted Assignments (CSV)",
                data=csv_buffer.getvalue(),
                file_name="submitted_assignments.csv",
                mime="text/csv",
                key="download_submitted_assignments_student"
            )
            if submitted_assignments:
                assignment_id = st.number_input("Assignment ID to View", min_value=1, step=1, key="view_assignment_id_student")
                if assignment_id:
                    with Database() as db:
                        db.c.execute('SELECT question, subject FROM assignments WHERE id = ?', (assignment_id,))
                        assignment = db.c.fetchone()
                        if assignment:
                            question, subject = assignment
                            st.text_area("❓ Question", value=question, height=100, disabled=True, key=f"question_student_view_{assignment_id}")

    # Evaluation history
    with st.expander("📜 Evaluation History", expanded=True):
        submissions = get_submissions(user['id'], user['role'])
        if submissions:
            history = []
            for sub in submissions:
                history_entry = {
                    'Timestamp': sub[4],
                    'Submission ID': sub[0],
                    'Subject': sub[1],
                    'Question': sub[2],
                    'Student': sub[7],
                    'Student ID': sub[9],
                    'Assignment ID': sub[10],
                    'Answer': sub[3],
                    'Status': sub[5],
                    'Deadline': sub[8],
                    'Similarity Score': 'N/A',
                    'Grammar Score': 'N/A',
                    'Keyword Score': 'N/A',
                    'Final Score': 'N/A',
                    'Suggestions': 'N/A'
                }
                if sub[6]:  # evaluation_result exists
                    try:
                        result = json.loads(sub[6])
                        history_entry.update({
                            'Similarity Score': f"{result.get('Similarity (%)', 0.0):.1f}% ({result.get('Similarity Score', 0)}/10)",
                            'Grammar Score': f"{result.get('Grammar Score', 0)}/10",
                            'Keyword Score': f"{result.get('Keyword Score', 0)}/10",
                            'Final Score': f"{result.get('Final Score', 0.0)}/10",
                            'Suggestions': result.get('Suggestions', 'N/A')
                        })
                    except json.JSONDecodeError:
                        history_entry['Similarity Score'] = 'Invalid data'
                        history_entry['Grammar Score'] = 'Invalid data'
                        history_entry['Keyword Score'] = 'Invalid data'
                        history_entry['Final Score'] = 'Invalid data'
                        history_entry['Suggestions'] = 'Invalid data'
                history.append(history_entry)
            history_df = pd.DataFrame(history)
            st.dataframe(history_df, use_container_width=True)

            # View Details for specific submission
            submission_id = st.number_input("Enter Submission ID to View Details", min_value=1, step=1, key="view_submission_id_history")
            if st.button("View Details", key="view_details_button_history"):
                view_details(submission_id)

            csv_buffer = io.StringIO()
            history_df.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_NONNUMERIC, date_format='%d/%m/%Y')
            st.download_button(
                label="📑 Download History (CSV)",
                data=csv_buffer.getvalue(),
                file_name="evaluation_history.csv",
                mime="text/csv",
                key="download_csv_history"
            )
        else:
            st.info("No submissions or evaluations yet.")

    # Detailed view section
    if st.session_state.selected_submission_id:
        with st.expander(f"Details for Submission ID {st.session_state.selected_submission_id}", expanded=True):
            with Database() as db:
                db.c.execute('''
                    SELECT s.answer, a.question, s.evaluation_result, a.model_answer, a.keywords, a.subject, a.deadline, s.student_id, s.assignment_id, u.username
                    FROM submissions s
                    JOIN assignments a ON s.assignment_id = a.id
                    JOIN users u ON s.student_id = u.id
                    WHERE s.id = ?
                ''', (st.session_state.selected_submission_id,))
                submission = db.c.fetchone()
                if submission:
                    student_answer, question, evaluation_result, model_answer, keywords_str, subject, deadline, student_id, assignment_id, student_username = submission
                    result = json.loads(evaluation_result) if evaluation_result else {}
                    keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()] if keywords_str else []
                    formatted_deadline = None
                    if deadline:
                        try:
                            deadline_dt = dt.datetime.strptime(deadline, '%Y-%m-%d %H:%M:%S')
                            formatted_deadline = deadline_dt.strftime('%d/%m/%Y')
                        except ValueError:
                            formatted_deadline = deadline
                    st.text_area("❓ Question", value=question, height=100, disabled=True, key=f"question_detail_{st.session_state.selected_submission_id}")
                    st.text_area("🧑‍🎓 Student's Answer", value=student_answer, height=150, disabled=True, key=f"student_answer_detail_{st.session_state.selected_submission_id}")
                    st.text_area("✍️ Model Answer", value=model_answer, height=150, disabled=True, key=f"model_answer_detail_{st.session_state.selected_submission_id}")
                    st.text_input("🔑 Keywords", value=", ".join(keywords), disabled=True, key=f"keywords_detail_{st.session_state.selected_submission_id}")
                    st.text_input("📅 Deadline", value=formatted_deadline or "None", disabled=True, key=f"deadline_detail_{st.session_state.selected_submission_id}")
                    st.text_input("🧑‍🎓 Student", value=f"{student_username} (ID: {student_id})", disabled=True, key=f"student_detail_{st.session_state.selected_submission_id}")
                    st.text_input("📋 Assignment ID", value=str(assignment_id), disabled=True, key=f"assignment_id_detail_{st.session_state.selected_submission_id}")
                    if result:
                        st.write(f"**Similarity Score:** {result.get('Similarity Score', 0)}/10 ({result.get('Similarity (%)', 0.0):.1f}%) - {result.get('Similarity Feedback', '')}")
                        st.write(f"**Grammar Score:** {result.get('Grammar Score', 0)}/10 - {result.get('Grammar Feedback', '')}")
                        if result.get('Grammar Matches'):
                            st.markdown("#### Grammar Issues (Top 5)")
                            for match in result.get('Grammar Matches', [])[:5]:
                                st.write(f"- {match['ruleId']}: {match['message']} (at: {match['context']})")
                        st.write(f"**Keyword Score:** {result.get('Keyword Score', 0)}/10 - {result.get('Keyword Feedback', '')}")
                        st.write(f"**Matched Keywords:** {result.get('Matched Keywords', 'None')}")
                        st.write(f"**Final Score:** {result.get('Final Score', 0.0)}/10")
                        if result.get('Suggestions'):
                            st.write(f"**Suggestions:** {result.get('Suggestions', 'None')}")
                    if st.button("Close Details", key=f"close_details_{st.session_state.selected_submission_id}"):
                        st.session_state.selected_submission_id = None
                        st.rerun()

st.markdown("---")
