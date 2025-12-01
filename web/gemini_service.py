"""
Gemini API service for web search and video search functionality.
Uses Google's Gemini Flash model for intelligent search.
"""

import requests
import json
from typing import List, Dict, Any

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyB5eriR4hgJd97KDCY-Bvb5ujPRW-E4N0Q"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def call_gemini_api(prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
    """
    Call Gemini API with a prompt.
    
    Args:
        prompt: The prompt to send to Gemini
        max_tokens: Maximum tokens in response
        
    Returns:
        Dictionary with success status and response/error
    """
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'candidates' in data and len(data['candidates']) > 0:
                text = data['candidates'][0]['content']['parts'][0]['text']
                return {'success': True, 'response': text}
            return {'success': False, 'error': 'No response from Gemini'}
        else:
            return {'success': False, 'error': f'API error: {response.status_code} - {response.text}'}
            
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timed out'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def search_web(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search the web for information on a topic using Gemini.
    Gemini will provide summarized web search results.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Dictionary with search results
    """
    prompt = f"""You are a helpful research assistant. Search and provide information about the following topic:

Topic: {query}

Please provide:
1. A brief overview/summary of the topic
2. {num_results} key points or facts about this topic
3. Relevant concepts that students should understand
4. Any important formulas, algorithms, or definitions if applicable

Format your response as JSON with this structure:
{{
    "summary": "Brief overview of the topic",
    "key_points": [
        {{"title": "Point 1", "description": "Description of point 1"}},
        {{"title": "Point 2", "description": "Description of point 2"}}
    ],
    "concepts": ["concept1", "concept2", "concept3"],
    "definitions": [
        {{"term": "Term 1", "definition": "Definition of term 1"}}
    ],
    "related_topics": ["related topic 1", "related topic 2"]
}}

Provide accurate, educational content suitable for computer science students.
Return ONLY valid JSON, no markdown or explanation."""

    result = call_gemini_api(prompt)
    
    if result['success']:
        try:
            # Try to parse the JSON response
            response_text = result['response'].strip()
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            search_results = json.loads(response_text)
            return {'success': True, 'results': search_results, 'query': query}
        except json.JSONDecodeError:
            # Return raw text if JSON parsing fails
            return {'success': True, 'results': {'summary': result['response']}, 'query': query}
    
    return result


def search_videos(topic: str, num_videos: int = 5) -> Dict[str, Any]:
    """
    Find educational video recommendations on a topic using Gemini.
    
    Args:
        topic: Topic to search videos for
        num_videos: Number of video recommendations
        
    Returns:
        Dictionary with video recommendations
    """
    prompt = f"""You are an educational content curator. Recommend {num_videos} educational videos or video series about the following computer science topic:

Topic: {topic}

For each video recommendation, provide:
1. A suggested video title (that would exist on YouTube or educational platforms)
2. What concepts the video would cover
3. Estimated duration (short: <10min, medium: 10-30min, long: >30min)
4. Difficulty level (beginner, intermediate, advanced)
5. A YouTube search query to find similar videos

Format your response as JSON:
{{
    "topic": "{topic}",
    "video_recommendations": [
        {{
            "title": "Video title",
            "platform": "YouTube",
            "concepts_covered": ["concept1", "concept2"],
            "duration": "medium",
            "difficulty": "intermediate",
            "search_query": "search query for youtube",
            "youtube_search_url": "https://www.youtube.com/results?search_query=encoded+search+query"
        }}
    ],
    "channels_to_follow": [
        {{"name": "Channel Name", "description": "Why this channel is good"}}
    ],
    "learning_path": "Suggested order to watch these topics"
}}

Focus on well-known educational channels like: MIT OpenCourseWare, Abdul Bari, mycodeschool, freeCodeCamp, CS Dojo, etc.
Return ONLY valid JSON, no markdown or explanation."""

    result = call_gemini_api(prompt)
    
    if result['success']:
        try:
            response_text = result['response'].strip()
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            video_results = json.loads(response_text)
            return {'success': True, 'results': video_results, 'topic': topic}
        except json.JSONDecodeError:
            return {'success': True, 'results': {'recommendations': result['response']}, 'topic': topic}
    
    return result


def generate_mcq_questions(topic: str, num_questions: int = 5, difficulty: str = "mixed") -> Dict[str, Any]:
    """
    Generate MCQ questions on a topic using Gemini.
    
    Args:
        topic: Topic to generate questions about
        num_questions: Number of questions to generate
        difficulty: easy, medium, hard, or mixed
        
    Returns:
        Dictionary with generated questions
    """
    prompt = f"""You are an expert educator creating multiple choice questions for a computer science course.

Topic: {topic}
Number of Questions: {num_questions}
Difficulty: {difficulty}

Generate {num_questions} high-quality multiple choice questions with:
- Clear, unambiguous question text
- Exactly 4 options (A, B, C, D)
- One correct answer
- Brief explanation for the correct answer

Format your response as JSON:
{{
    "topic": "{topic}",
    "questions": [
        {{
            "id": 1,
            "question": "What is the time complexity of binary search?",
            "options": {{
                "A": "O(1)",
                "B": "O(n)",
                "C": "O(log n)",
                "D": "O(n log n)"
            }},
            "correct_answer": "C",
            "explanation": "Binary search divides the search space in half each iteration, resulting in logarithmic time complexity.",
            "difficulty": "medium"
        }}
    ]
}}

Return ONLY valid JSON, no markdown or explanation."""

    result = call_gemini_api(prompt, max_tokens=4096)
    
    if result['success']:
        try:
            response_text = result['response'].strip()
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            questions = json.loads(response_text)
            return {'success': True, 'data': questions}
        except json.JSONDecodeError:
            return {'success': False, 'error': 'Failed to parse questions', 'raw': result['response']}
    
    return result


def explain_topic(topic: str, detail_level: str = "comprehensive") -> Dict[str, Any]:
    """
    Get a detailed explanation of a topic using Gemini.
    
    Args:
        topic: Topic to explain
        detail_level: brief, moderate, or comprehensive
        
    Returns:
        Dictionary with explanation
    """
    detail_instructions = {
        "brief": "Provide a brief 2-3 paragraph explanation",
        "moderate": "Provide a moderate explanation with key points and examples",
        "comprehensive": "Provide a comprehensive explanation with examples, use cases, and related concepts"
    }
    
    prompt = f"""You are a computer science professor explaining a topic to students.

Topic: {topic}

{detail_instructions.get(detail_level, detail_instructions['comprehensive'])}

Include:
1. Definition and overview
2. Key concepts
3. Examples or use cases
4. Common misconceptions
5. Related topics to explore

Format your response as JSON:
{{
    "topic": "{topic}",
    "definition": "Clear definition",
    "overview": "Brief overview",
    "key_concepts": [
        {{"concept": "Concept name", "explanation": "Brief explanation"}}
    ],
    "examples": [
        {{"title": "Example title", "description": "Example description", "code": "optional code snippet"}}
    ],
    "misconceptions": ["Common misconception 1"],
    "related_topics": ["Related topic 1", "Related topic 2"],
    "difficulty_level": "beginner|intermediate|advanced"
}}

Return ONLY valid JSON, no markdown or explanation."""

    result = call_gemini_api(prompt, max_tokens=4096)
    
    if result['success']:
        try:
            response_text = result['response'].strip()
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            explanation = json.loads(response_text)
            return {'success': True, 'data': explanation}
        except json.JSONDecodeError:
            return {'success': True, 'data': {'explanation': result['response']}}
    
    return result
