from __future__ import annotations
import os
import json
import argparse
import requests
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# RAG Imports
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")


class SubTask(BaseModel):
    """Represents a single sub-task"""
    task: str = Field(description="The sub-task description")


class AcceptanceCriterion(BaseModel):
    """Represents a single acceptance criterion"""
    criterion: str = Field(description="The acceptance criterion description")


class UserStory(BaseModel):
    """Represents a complete user story with acceptance criteria and sub-tasks"""
    user_story: str = Field(description="The user story in format: As a [user], I want [intent], so that [value].")
    acceptance_criteria: List[AcceptanceCriterion] = Field(description="List of acceptance criteria")
    sub_tasks: List[SubTask] = Field(description="List of sub-tasks")


class UserStoryCollection(BaseModel):
    """Collection of user stories generated from a transcript"""
    stories: List[UserStory] = Field(description="List of user stories extracted from the transcript")


def load_system_prompt() -> str:
    """Load the system prompt from PromptTemplate.txt"""
    prompt_path = os.path.join(os.path.dirname(__file__), "PromptTemplate.txt")
    try:
        with open(prompt_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"PromptTemplate.txt not found at {prompt_path}. "
            "Please ensure the file exists in the same directory as this script."
        )


def create_trello_card(user_story: UserStory, list_id: str) -> Optional[str]:
    """
    Creates a Trello card with the user story as the name and 
    acceptance criteria + sub-tasks as the description.
    
    Args:
        user_story: The UserStory object
        list_id: The Trello list ID where the card should be created
        
    Returns:
        The card ID if successful, None otherwise
    """
    if not TRELLO_API_KEY or not TRELLO_TOKEN:
        print("Warning: TRELLO_API_KEY and TRELLO_TOKEN must be set for Trello integration.")
        return None
    
    # Format the description with acceptance criteria only
    description = "## Acceptance Criteria\n\n"
    for criterion in user_story.acceptance_criteria:
        description += f"- {criterion.criterion}\n"

    
    url = "https://api.trello.com/1/cards"
    query = {
        'idList': list_id,
        'key': TRELLO_API_KEY,
        'token': TRELLO_TOKEN,
        'name': user_story.user_story,
        'desc': description
    }
    
    try:
        response = requests.post(url, params=query)
        if response.status_code == 200:
            card_id = response.json().get('id')
            if not card_id:
                print(f"❌ Error: Card created but no ID returned")
                return None
                
            print(f"✅ Created Trello card: {user_story.user_story[:40]}...")
            
            # --- Native Checklist Creation ---
            # Create a checklist named "Sub-Tasks"
            checklist_url = f"https://api.trello.com/1/cards/{card_id}/checklists"
            checklist_query = {
                'key': TRELLO_API_KEY,
                'token': TRELLO_TOKEN,
                'name': 'Sub-Tasks'
            }
            
            checklist_resp = requests.post(checklist_url, params=checklist_query)
            if checklist_resp.status_code == 200:
                checklist_id = checklist_resp.json().get('id')
                
                # Add items to the checklist
                for task in user_story.sub_tasks:
                    item_url = f"https://api.trello.com/1/checklists/{checklist_id}/checkItems"
                    item_query = {
                        'key': TRELLO_API_KEY,
                        'token': TRELLO_TOKEN,
                        'name': task.task
                    }
                    requests.post(item_url, params=item_query)
                print(f"   ✅ Added checklist with {len(user_story.sub_tasks)} sub-tasks")
            else:
                print(f"   ⚠️ Could not create checklist: {checklist_resp.text}")
                
            return card_id
        else:
            print(f"❌ Error creating Trello card: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error creating Trello card: {e}")
        return None


def format_user_story_output(story: UserStory, index: int) -> str:
    """Format a single user story for console/file output"""
    output = f"\n{'='*80}\n"
    output += f"USER STORY #{index}\n"
    output += f"{'='*80}\n\n"
    
    output += f"**User Story:**\n{story.user_story}\n\n"
    
    output += "**Acceptance Criteria:**\n"
    for criterion in story.acceptance_criteria:
        output += f"- {criterion.criterion}\n"
    
    output += "\n**Sub-Tasks:**\n"
    for task in story.sub_tasks:
        output += f"- [ ] {task.task}\n"
    
    return output


def process_transcript(transcript: str, trello_list_id: Optional[str] = None, 
                       output_file: Optional[str] = None) -> UserStoryCollection:
    """
    Process a customer interview transcript and generate user stories.
    
    Args:
        transcript: The customer interview transcript text
        trello_list_id: Optional Trello list ID to create cards
        output_file: Optional file path to save the output
        
    Returns:
        UserStoryCollection containing all generated user stories
    """
    # Load system prompt
    system_prompt = load_system_prompt()
    
    # Initialize OpenAI model with structured output
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Bind the structured output schema
    structured_llm = llm.with_structured_output(UserStoryCollection)
    
    # Create messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Convert the following customer interview transcript into user stories:\n\n{transcript}")
    ]
    
    print("🔄 Processing transcript with OpenAI GPT-4o...")
    
    # Invoke the LLM
    result = structured_llm.invoke(messages)
    
    print(f"✅ Generated {len(result.stories)} user story(ies)\n")
    
    # Format output
    full_output = ""
    for i, story in enumerate(result.stories, 1):
        story_output = format_user_story_output(story, i)
        full_output += story_output
        print(story_output)
        
        # Create Trello card if list_id is provided
        if trello_list_id:
            create_trello_card(story, trello_list_id)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_output)
        print(f"\n💾 Output saved to: {output_file}")
    
    return result


class RAGPipeline:
    def __init__(self, text_content: str):
        print("🔄 Initializing RAG pipeline...")
        # 1. Chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.create_documents([text_content])
        
        # 2. Embedding Model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        
        # 3. Vector Store
        self.vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        # 4. LLM
        self.llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        print("✅ RAG pipeline ready.")

    def ask(self, query: str) -> str:
        # Retrieve docs
        docs = self.vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"Context:\n{context}"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

def setup_rag_pipeline(text_content: str) -> RAGPipeline:
    return RAGPipeline(text_content)



def main():
    parser = argparse.ArgumentParser(
        description="Convert customer interview transcripts into Agile user stories with Trello integration."
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Path to transcript file'
    )
    input_group.add_argument(
        '--text', '-t',
        type=str,
        help='Direct transcript text'
    )
    
    # Trello options
    parser.add_argument(
        '--trello-list-id', '-l',
        type=str,
        help='Trello list ID to create cards (optional)'
    )
    parser.add_argument(
        '--no-trello',
        action='store_true',
        help='Disable Trello integration even if credentials are set'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path to save user stories (optional)'
    )
    
    # RAG Options
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Ask a specific question about the transcript'
    )
    parser.add_argument(
        '--chat',
        action='store_true',
        help='Enter interactive chat mode with the transcript'
    )
    
    args = parser.parse_args()
    
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        return
    
    # Get transcript text
    if args.input:
        try:
            with open(args.input, 'r') as f:
                transcript = f.read()
            print(f"📄 Loaded transcript from: {args.input}\n")
        except FileNotFoundError:
            print(f"❌ Error: File not found: {args.input}")
            return
    else:
        transcript = args.text
        print("📝 Processing direct text input\n")
    
    # RAG Workflow
    if args.query or args.chat:
        rag = setup_rag_pipeline(transcript)
        
        if args.query:
            print(f"\n❓ Question: {args.query}")
            # response = rag_chain.invoke({"input": args.query})
            # print(f"💡 Answer: {response['answer']}\n")
            answer = rag.ask(args.query)
            print(f"💡 Answer: {answer}\n")
            
        if args.chat:
            print("\n💬 Entering chat mode (type 'exit' or 'quit' to stop):")
            while True:
                user_input = input("\n> ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                # response = rag_chain.invoke({"input": user_input})
                # print(f"💡 Answer: {response['answer']}")
                answer = rag.ask(user_input)
                print(f"💡 Answer: {answer}")
        return  # Exit after RAG operations if either was chosen

    # Standard User Story Workflow

    # Determine Trello list ID
    trello_list_id = None
    if not args.no_trello:
        trello_list_id = args.trello_list_id or os.getenv("TRELLO_LIST_ID")
    
    # Process the transcript
    try:
        process_transcript(transcript, trello_list_id, args.output)
        print("\n✅ Processing complete!")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
