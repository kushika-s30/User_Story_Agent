from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import user_story_agent
import os
import json
from typing import Optional
import io

# Import Parsing Libraries
import pypdf
import pandas as pd
# check for python-docx
try:
    from docx import Document
except ImportError:
    pass # Handle gracefully if not found

app = FastAPI()

# Mount static files to serve index.html
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

def parse_file(file: UploadFile) -> str:
    """Extracts text from various file formats."""
    content = ""
    filename = file.filename.lower()
    
    try:
        content_bytes = file.file.read()
        file.file.seek(0)
        
        if filename.endswith(".txt"):
            content = content_bytes.decode("utf-8")
            
        elif filename.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            text_list = []
            for page in reader.pages:
                text_list.append(page.extract_text())
            content = "\\n".join(text_list)
            
        elif filename.endswith(".docx"):
            # python-docx requires a file-like object
            doc = Document(io.BytesIO(content_bytes))
            content = "\\n".join([para.text for para in doc.paragraphs])
            
        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content_bytes))
            content = df.to_string()
            
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(content_bytes))
            content = df.to_string()
            
        else:
            raise ValueError(f"Unsupported file format: {filename}")
            
    except Exception as e:
        print(f"Error parsing file: {e}")
        # Only raise parsing errors, otherwise return empty so user is notified
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
        
    return content

@app.post("/api/craft-stories")
async def craft_stories(
    transcript: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    mock_mode: str = Form("false") # Receive as string to handle FormData quirk
):
    # Safe boolean conversion
    is_mock = str(mock_mode).lower() in ("true", "1", "yes", "on")
    
    final_transcript = ""

    # 1. Handle File Upload
    if file:
        print(f"📂 Processing file: {file.filename}")
        parsed_content = parse_file(file)
        if parsed_content:
            final_transcript = parsed_content
            
    # 2. Handle Text Input (Fallback or Override)
    if transcript:
         # If both exist, append transcript text.
         if final_transcript:
             final_transcript += "\\n\\n" + transcript
         else:
             final_transcript = transcript

    # 3. Validation
    if not final_transcript and not is_mock:
        raise HTTPException(status_code=400, detail="Please provide a transcript text or upload a file.")

    # Get Default Trello List ID from env (needed for checking availability)
    trello_list_id = os.getenv("TRELLO_LIST_ID")

    if is_mock:
        print("🧪 Mock Mode enabled: Generating dummy stories...")
        # Create dummy data using the imported classes
        dummy_story = user_story_agent.UserStory(
            user_story="As a tester, I want to use mock data, so that I save OpenAI tokens.",
            acceptance_criteria=[
                user_story_agent.AcceptanceCriterion(criterion="No API calls to OpenAI are made"),
                user_story_agent.AcceptanceCriterion(criterion="Stories are still formatted correctly"),
                user_story_agent.AcceptanceCriterion(criterion="Trello cards are created if enabled")
            ],
            sub_tasks=[
                user_story_agent.SubTask(task="Verify UI rendering"),
                user_story_agent.SubTask(task="Check Trello for new card")
            ]
        )
        stories = [dummy_story]
        
        stories_data = [story.model_dump() for story in stories]
        return JSONResponse(content={"stories": stories_data, "trello_synced": False})
    
    try:
        # Real processing
        result = user_story_agent.process_transcript(final_transcript, trello_list_id=None)
        
        # Convert Pydantic models to dict for JSON response
        stories_data = [story.model_dump() for story in result.stories]
        
        # Return stories without sync status
        return JSONResponse(content={"stories": stories_data, "trello_synced": False})
        
    except Exception as e:
        print(f"Error processing transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class SyncStoryRequest(BaseModel):
    user_story: str
    acceptance_criteria: list[str]
    sub_tasks: list[str]

@app.post("/api/sync-story")
async def sync_story(story: SyncStoryRequest):
    trello_list_id = os.getenv("TRELLO_LIST_ID")
    if not trello_list_id:
        raise HTTPException(status_code=500, detail="TRELLO_LIST_ID not set in server environment")

    # Reconstruct the Pydantic object expected by user_story_agent
    user_story_obj = user_story_agent.UserStory(
        user_story=story.user_story,
        acceptance_criteria=[user_story_agent.AcceptanceCriterion(criterion=c) for c in story.acceptance_criteria],
        sub_tasks=[user_story_agent.SubTask(task=t) for t in story.sub_tasks]
    )

    card_id = user_story_agent.create_trello_card(user_story_obj, trello_list_id)
    
    if card_id:
        return {"success": True, "card_id": card_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to create Trello card")

@app.post("/api/upload-context")
async def upload_context(file: UploadFile = File(...)):
    """Uploads a file to the context_docs directory."""
    context_dir = "context_docs"
    if not os.path.exists(context_dir):
        os.makedirs(context_dir)
        
    try:
        # Save raw file first
        file_path = os.path.join(context_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
            
        return {"success": True, "filename": file.filename, "message": "Context file uploaded successfully"}
    except Exception as e:
        print(f"Error uploading context file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload context file: {str(e)}")

class AskRequest(BaseModel):
    query: str

@app.post("/api/ask")
async def ask_rag(request: AskRequest):
    """Answers a question using RAG (Transcript + Context Docs)."""
    try:
        # For simplicity, we'll re-initialize the pipeline on each request.
        # In production, you'd cache the vector store.
        
        # 1. Gather all context
        combined_text = ""
        
        # Add Transcript if previously uploaded/processed (Not trivial to persist state unless we save it)
        # For this demo, let's just use the context docs as the primary source if no transcript is active?
        # OR: We should probably save the last processed transcript to a file like `current_transcript.txt`.
        
        # Let's check for context docs
        context_dir = "context_docs"
        if os.path.exists(context_dir):
            for filename in os.listdir(context_dir):
                file_path = os.path.join(context_dir, filename)
                # Parse based on extension using existing logic?
                # For now, let's assume we simply read text files, or try to use user_story_agent's loading.
                # To keep it robust, let's reuse parse_file logic here or in agent.
                
                # ...Actually, app.py has parse_file. Let's use it.
                # Problem: parse_file expects UploadFile.
                # Let's just read raw text files for now as per plan, 
                # OR refactor parse_file. Let's keep it simple: support .txt in context for now.
                if filename.lower().endswith(".txt"):
                    with open(file_path, "r") as f:
                        combined_text += f"\n\n--- Context: {filename} ---\n" + f.read()

        if not combined_text.strip():
             return {"answer": "No context documents found to answer from. Please upload context."}

        # Initialize RAG
        rag = user_story_agent.setup_rag_pipeline(combined_text)
        answer = rag.ask(request.query)
        
        return {"answer": answer}
        
    except Exception as e:
        print(f"Error in RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
