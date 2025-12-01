# Smart Source Prioritization & Teachers AQG Feature

## Overview

This update adds two major features to the BanglaRAG chatbot system:

1. **Smart Source Prioritization** - Intelligently routes questions to the right database
2. **Automatic Question Generation (AQG)** - Teachers can generate assessment questions from course content

---

## 🎯 Feature 1: Smart Source Prioritization

### How It Works

The system now intelligently prioritizes database sources based on question type:

#### Course-Specific Questions (Prioritize Course Database)

Keywords detected: `module`, `syllabus`, `prerequisite`, `instructor`, `course`, `assignment`, `exam`, `grade`, `lecture`, `schedule`, `learning outcome`, `week`, `semester`, `section`

**Example Questions:**

- ✅ "What is module 1 of the course?"
- ✅ "What are the prerequisites?"
- ✅ "What topics are covered in the syllabus?"
- ✅ "What is covered in week 3?"

**Result:** Only searches course_materials database, cites course_knowledge_base.txt

#### Theory/Algorithm Questions (Search Both)

General computer science or algorithm theory questions

**Example Questions:**

- 🔍 "Explain quicksort algorithm"
- 🔍 "What is the master theorem?"
- 🔍 "Compare binary search tree vs hash table"

**Result:** Searches both databases, cites most relevant source (course text or Cormen PDF)

#### Fallback Behavior

1. If course-specific question → Check course DB first
2. If no results in course DB → Check PDF DB as fallback
3. If no results anywhere → Return empty (chatbot will indicate no information available)

### Code Implementation

Located in: `web/chatbot_api.py`

```python
def search_dual_databases(query: str, k: int = 3):
    """
    Search both course materials and PDF database with smart prioritization.
    """
    query_lower = query.lower()

    # Course-specific keywords
    course_keywords = [
        'module', 'syllabus', 'prerequisite', 'instructor', 'course',
        'assignment', 'exam', 'grade', 'lecture', 'schedule', 'learning outcome',
        'week', 'semester', 'section'
    ]

    is_course_specific = any(keyword in query_lower for keyword in course_keywords)

    if is_course_specific:
        # Prioritize course materials, use PDF as fallback only
        ...
    else:
        # Search both, merge by relevance
        ...
```

---

## 👨‍🏫 Feature 2: Automatic Question Generation (AQG)

### Access

Navigate to: **http://localhost:5000/teachers**

### Features

#### 1. Module Selection

- All Modules (searches entire course)
- MODULE 1: ARRAYS AND STRINGS
- MODULE 2: LINKED LISTS
- MODULE 3: STACKS AND QUEUES
- MODULE 4: TREES
- MODULE 5: SORTING AND SEARCHING
- MODULE 6: HASH TABLES

#### 2. Difficulty Levels

- **Easy**: Basic recall and understanding questions
- **Medium**: Application and analysis questions
- **Hard**: Advanced synthesis and evaluation questions
- **Mixed**: Combination of all difficulty levels

#### 3. Question Types (Multi-Select)

✓ **Multiple Choice** - 4 options (A, B, C, D) with correct answer
✓ **True/False** - Binary questions with reasoning
✓ **Short Answer** - 1-2 sentence responses required
✓ **Explain/Describe** - Detailed conceptual explanations

#### 4. Number of Questions

- Range: 1-20 questions per generation
- Default: 5 questions

### How to Use

1. **Select Configuration**

   - Choose module (specific or all)
   - Select difficulty level
   - Pick number of questions
   - Check desired question types

2. **Generate Questions**

   - Click "Generate Questions" button
   - System retrieves relevant course content
   - LLM generates structured questions with answers

3. **Review Results**

   - Questions appear in the right panel
   - Each shows: Module, Difficulty, Type
   - For multiple choice: 4 options displayed
   - Complete answers provided

4. **Export Questions**
   - Click "📥 Export Questions" button
   - Downloads formatted .txt file
   - Includes all metadata and answers
   - Ready for use in exams/quizzes

### API Endpoint

**POST** `/api/teachers/generate-questions`

**Request Body:**

```json
{
  "module": "MODULE 1", // or "all"
  "difficulty": "medium", // easy|medium|hard|mixed
  "num_questions": 5,
  "question_types": ["multiple-choice", "true-false", "short-answer", "explain"]
}
```

**Response:**

```json
{
  "success": true,
  "questions": [
    {
      "question": "What is the time complexity of...",
      "type": "multiple-choice",
      "options": ["O(1)", "O(n)", "O(log n)", "O(n²)"],
      "answer": "O(n)",
      "difficulty": "medium",
      "module": "MODULE 1"
    }
  ],
  "count": 5
}
```

### Code Files

- **Frontend**: `web/teachers.html` (550+ lines)

  - Clean dashboard UI
  - Form controls for all options
  - Real-time question display
  - Export functionality

- **Backend**: `web/chatbot_api.py`
  - Route: `@app.route("/teachers")` - Serves HTML page
  - API: `@app.route("/api/teachers/generate-questions")` - Generates questions
  - Uses dual database to retrieve course content
  - Sends structured prompt to LLM
  - Parses JSON response

---

## 🧪 Testing Guide

### Test Smart Prioritization

1. **Course-Specific Test**

   ```
   Question: "What is module 1 of the course?"
   Expected: Cites course_knowledge_base.txt
   Expected Content: MODULE 1: ARRAYS AND STRINGS details
   ```

2. **Theory Test**

   ```
   Question: "Explain the master theorem"
   Expected: Cites Cormen PDF (if content exists)
   Otherwise: Uses course materials
   ```

3. **Mixed Test**
   ```
   Question: "What sorting algorithms are in the syllabus?"
   Expected: Cites course_knowledge_base.txt (course-specific keyword "syllabus")
   ```

### Test Question Generation

1. **Navigate to Teachers Page**

   - Open: http://localhost:5000/teachers

2. **Generate Module-Specific Questions**

   - Select: MODULE 1
   - Difficulty: Medium
   - Count: 3
   - Types: Multiple Choice, Short Answer
   - Click Generate

3. **Verify Output**

   - Questions relate to MODULE 1 content (arrays, strings)
   - Difficulty matches selection
   - Question types match checkboxes
   - Answers are complete and correct

4. **Export Test**
   - Click "📥 Export Questions"
   - Verify .txt file downloads
   - Check formatting is readable

---

## 📊 Benefits

### For Students

- ✅ More accurate answers (right source for right question)
- ✅ Course structure questions answered from syllabus
- ✅ Theory questions get deep algorithm explanations from textbook

### For Teachers

- 🎯 Generate assessment questions in seconds
- 📚 Questions directly based on course content
- 🔄 Different difficulty levels for differentiated assessment
- 📝 Multiple question types for varied evaluation
- 💾 Easy export for exam/quiz creation
- ⚡ Saves hours of manual question writing

---

## 🚀 Quick Start

### Start Server

```bash
python .\web\chatbot_api.py
```

### Access Points

- **Student Chatbot**: http://localhost:5000/
- **Teachers Dashboard**: http://localhost:5000/teachers

### Test Endpoints

```bash
# Health check
curl http://localhost:5000/api/health

# Generate questions
curl -X POST http://localhost:5000/api/teachers/generate-questions \
  -H "Content-Type: application/json" \
  -d '{
    "module": "MODULE 1",
    "difficulty": "medium",
    "num_questions": 3,
    "question_types": ["multiple-choice"]
  }'
```

---

## 🔧 Configuration

### Modify Course Keywords

Edit `web/chatbot_api.py`, line ~110:

```python
course_keywords = [
    'module', 'syllabus', 'prerequisite', 'instructor', 'course',
    'assignment', 'exam', 'grade', 'lecture', 'schedule', 'learning outcome',
    'week', 'semester', 'section'
]
```

Add any domain-specific keywords that should prioritize course materials.

### Modify Question Types

Edit `web/teachers.html`, search for checkbox options:

```html
<label>
  <input type="checkbox" value="your-new-type" checked />
  Your New Type
</label>
```

---

## 📝 Future Enhancements

### Potential Improvements

- [ ] Add question difficulty slider (finer control)
- [ ] Save generated question sets to database
- [ ] Generate entire exam papers with point distribution
- [ ] Add question edit functionality before export
- [ ] Support multiple export formats (PDF, DOCX, JSON)
- [ ] Add Bloom's Taxonomy level selection
- [ ] Include rubrics/grading criteria in output
- [ ] Add question bank management (save/reuse questions)

---

## ✅ Completion Status

- [x] Smart source prioritization implemented
- [x] Course-specific keyword detection
- [x] Dual database fallback logic
- [x] Teachers dashboard UI created
- [x] Question generation API endpoint
- [x] Module selection functionality
- [x] Difficulty level controls
- [x] Multiple question type support
- [x] Export questions feature
- [x] Server endpoints updated
- [x] Documentation complete

**Status**: ✅ **FULLY OPERATIONAL**

---

## 📖 Related Documentation

- `web/DUAL_DATABASE_SUCCESS.md` - Dual database system overview
- `web/VOICE_MODE_GUIDE.md` - Voice input feature
- `web/DATABASE_FIX.md` - Database troubleshooting
- `README.md` - Main project documentation
