# Subjective-Answers-Evaluation-Using-ML-and-NLP# 📘 Subjective Answers Evaluation Using ML and NLP

This project is a **Streamlit-based web application** that automates the evaluation of subjective answers written by students. It utilizes **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques to assess semantic similarity, grammar, keyword usage, and answer quality. Designed for educational institutions, it supports role-based access for **Admins**, **Teachers**, and **Students**.

---

## 🔍 Project Overview

### 🎯 Objective:
To reduce manual effort in evaluating long-form answers by automating the process using modern NLP models, while maintaining fairness, consistency, and accuracy in scoring.

### 🧠 Key Functionalities:
- **Semantic Similarity Scoring** using pre-trained transformers.
- **Grammar Quality Evaluation** via LanguageTool.
- **Keyword Matching** with synonym recognition (WordNet).
- **Answer Length Penalty** for too short or excessively long responses.
- **Final Score Aggregation** and feedback generation.
- **Role-Based Access** (Admin/Teacher/Student).
- **User Management, Approvals, and Status Control**.
- **Modern Dark Mode UI** using Tailwind CSS in Streamlit.

---

## 🏗️ System Architecture

The system is structured into multiple layers:

- **Frontend**: Built using `Streamlit`, styled with custom CSS and Tailwind-based themes.
- **Backend**: 
  - Uses `sqlite3` for database management.
  - NLP via `nltk`, `sentence-transformers`, `language-tool-python`.
- **Storage**: `SQLite` database with 3 main tables: `users`, `assignments`, and `submissions`.

---

## 📂 Project Structure
Subjective-Answers-Evaluation-Using-ML-and-NLP/
│
├── app1.py # Main application file (Streamlit app)
├── test.py # Secondary/test version of the app
├── requirements.txt # Python dependencies
├── README.md # This documentation file
└── evaluator.db # Auto-generated SQLite database (after first run)


---

## 👥 User Roles

### 🛡️ Admin:
- Approve or deactivate users.
- Monitor all submissions.
- View system-wide analytics and download evaluation data.

### 👨‍🏫 Teacher:
- Create assignments with:
  - Subject
  - Question
  - Model answer
  - Keywords
- Evaluate student responses.
- Upload assignments via CSV in bulk.

### 🧑‍🎓 Student:
- View available assignments.
- Submit textual answers.
- View feedback and evaluation scores.

---

## 🧠 Evaluation Criteria

The submitted answers are evaluated based on the following:

| Component           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Semantic Similarity** | How closely the student answer aligns semantically with the model answer. |
| **Grammar Quality**     | Assessed using LanguageTool for grammatical errors.                      |
| **Keyword Matching**    | Compares answer keywords, considers synonyms using WordNet.              |
| **Length Penalty**      | Penalizes very short (<15 words) or overly long (>200 words) answers.    |
| **Final Score**         | Weighted average of all above scores, normalized to 10.                  |
| **Feedback**            | Textual guidance on how to improve the answer.                           |

---


