import argparse
import json
import os
import base64
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Try to import the new Google Gen AI SDK, fallback to OpenAI if not available
try:
    from google import genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("⚠️  Google Gen AI SDK not found. Install with: pip install google-genai")
    print("   Falling back to OpenAI for fact-checking.")

# Constants
FACT_CHECK_STATE_FILE = "fact_check_state.json"
CONTROL_LOG_FILE = "fact_check_control_log.json"
EXTRACTED_DIR = "extracted_pages"
SMALL_MODEL = "gpt-4.1-mini"  # Fallback for fact-checking if Google not available
GEMINI_MODEL = "gemini-2.5-flash-lite-preview-06-17"  # Google's Flash-Lite preview model
DEFAULT_FINAL_DOCUMENT = "book_analysis_final.md"

load_dotenv()

# Initialize clients
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in the .env file")

openai_client = OpenAI(api_key=openai_api_key)

# Initialize Google client if available
google_client = None
if GOOGLE_AVAILABLE and google_api_key:
    try:
        google_client = genai.Client(api_key=google_api_key)
        print("✅ Google Gen AI SDK initialized with Gemini 2.5 Flash-Lite for fact-checking")
    except Exception as e:
        print(f"⚠️  Failed to initialize Google client: {e}")
        print("   Falling back to OpenAI for fact-checking.")
elif not google_api_key:
    print("⚠️  GOOGLE_API_KEY not found in .env file")
    print("   Add GOOGLE_API_KEY to use Gemini 2.5 Flash-Lite for faster fact-checking.")

def load_final_document(document_path: str) -> str:
    """Load the final analysis document for fact-checking."""
    try:
        with open(document_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Final document not found: {document_path}")

def initialize_fact_check_state() -> Dict[str, Any]:
    """Initialize or load the fact-check state JSON."""
    if os.path.exists(FACT_CHECK_STATE_FILE):
        with open(FACT_CHECK_STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "current_page": 1,
        "total_pages": 0,
        "control_log": [],
        "phase": "fact_checking",
        "started_at": datetime.now().isoformat(),
        "completed_pages": 0
    }

def save_fact_check_state(state: Dict[str, Any]):
    """Save the fact-check state to JSON."""
    with open(FACT_CHECK_STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def save_control_log(control_log: List[Dict[str, Any]]):
    """Save the control log with fact-checking results."""
    log_data = {
        "generated_at": datetime.now().isoformat(),
        "total_checks": len(control_log),
        "issues_found": len([entry for entry in control_log if entry.get("issues_found", False)]),
        "entries": control_log
    }
    
    with open(CONTROL_LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=4)

def extract_page_content(pdf_path: str, page_num: int) -> Dict[str, Any]:
    """Extract text and get image path for a specific page."""
    img_path = Path(EXTRACTED_DIR) / f"page_{page_num}.png"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Page image not found: {img_path}")
    
    # Extract text from PDF page
    with pdfplumber.open(pdf_path) as pdf:
        if page_num > len(pdf.pages):
            raise ValueError(f"Page {page_num} does not exist in PDF")
        
        page = pdf.pages[page_num - 1]  # 0-indexed
        text = page.extract_text() or ""
    
    return {
        "text": text,
        "image_path": str(img_path)
    }

def fact_check_page_against_analysis(page_num: int, page_text: str, img_path: str, final_analysis: str) -> Dict[str, Any]:
    """
    Fact-check a specific page against the final analysis document.
    Returns a dictionary with fact-checking results.
    """
    try:
        # Skip if page has no meaningful content
        if not page_text or len(page_text.strip()) < 10:
            return {
                "page_number": page_num,
                "status": "skipped",
                "reason": "minimal_content",
                "issues_found": False,
                "discrepancies": [],
                "confidence": "N/A",
                "checked_at": datetime.now().isoformat()
            }
        
        prompt = f"""You are performing TARGETED FACT-CHECKING to ensure the final analysis accurately represents the original book's key concepts and maintains factual integrity for its intended purpose.

CONTEXT: This is an analysis document, NOT a reproduction of the book. The goal is to capture essential insights and actionable frameworks, not every minor detail.

TASK: Compare this page's content with the final analysis to identify ONLY significant issues that matter for the analysis's credibility and completeness.

ORIGINAL BOOK PAGE {page_num} CONTENT:
{page_text}

FINAL ANALYSIS DOCUMENT:
{final_analysis}

FOCUS ON THESE TYPES OF ISSUES:
1. **Major factual errors** - Incorrect numbers, misattributed quotes, wrong frameworks
2. **Significant misrepresentations** - Core concepts explained incorrectly or context lost
3. **Important missing frameworks** - Key actionable strategies or frameworks that would strengthen the analysis
4. **Critical omissions** - Essential concepts that significantly impact the main arguments

DO NOT FLAG:
- Minor details or examples that don't affect core understanding
- Slight paraphrasing or condensing of content
- Omission of supporting anecdotes or secondary examples
- Different organizational structure or presentation order
- Minor contextual details that don't impact actionable insights

RESPOND WITH A JSON object with these fields:
- "issues_found": boolean (true ONLY if significant problems detected)
- "discrepancies": array of strings describing ONLY major factual errors
- "missing_content": array of strings for IMPORTANT frameworks/concepts that would strengthen the analysis
- "misrepresentations": array of strings for CORE concepts that are incorrectly represented
- "confidence": string ("high", "medium", "low") - your confidence in this assessment
- "summary": string - brief summary focusing on significant issues only

Return ONLY the JSON object, no other text."""

        # Try Google Gemini 2.5 Flash-Lite first (fastest and cheapest)
        if google_client:
            try:
                response = google_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[prompt]
                )
                
                if response and response.text:
                    # Parse JSON response
                    result_text = response.text.strip()
                    # Remove markdown code blocks if present
                    if result_text.startswith("```json"):
                        result_text = result_text[7:-3]
                    elif result_text.startswith("```"):
                        result_text = result_text[3:-3]
                    
                    try:
                        fact_check_result = json.loads(result_text)
                        fact_check_result.update({
                            "page_number": page_num,
                            "status": "completed",
                            "method": "gemini",
                            "checked_at": datetime.now().isoformat()
                        })
                        return fact_check_result
                    except json.JSONDecodeError:
                        print(f"    ⚠️  Invalid JSON from Gemini for page {page_num}, trying OpenAI...")
                else:
                    print(f"    ⚠️  Empty response from Gemini for page {page_num}, trying OpenAI...")
            except Exception as e:
                print(f"    ⚠️  Gemini error for page {page_num}: {str(e)[:100]}... Trying OpenAI...")
        
        # Fallback to OpenAI
        with open(img_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        
        response = openai_client.chat.completions.create(
            model=SMALL_MODEL,
            messages=[
                {"role": "system", "content": "You are a fact-checker ensuring analysis documents accurately represent source material. Return only valid JSON as specified."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ]
        )
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            result_text = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            elif result_text.startswith("```"):
                result_text = result_text[3:-3]
            
            try:
                fact_check_result = json.loads(result_text)
                fact_check_result.update({
                    "page_number": page_num,
                    "status": "completed",
                    "method": "openai",
                    "checked_at": datetime.now().isoformat()
                })
                return fact_check_result
            except json.JSONDecodeError:
                print(f"    ❌ Invalid JSON from OpenAI for page {page_num}")
                return {
                    "page_number": page_num,
                    "status": "error",
                    "reason": "invalid_json_response",
                    "issues_found": False,
                    "discrepancies": [],
                    "confidence": "low",
                    "checked_at": datetime.now().isoformat()
                }
        else:
            print(f"    ❌ Empty response from OpenAI for page {page_num}")
            return {
                "page_number": page_num,
                "status": "error",
                "reason": "empty_response",
                "issues_found": False,
                "discrepancies": [],
                "confidence": "low",
                "checked_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"    ❌ Error fact-checking page {page_num}: {str(e)}")
        return {
            "page_number": page_num,
            "status": "error",
            "reason": str(e),
            "issues_found": False,
            "discrepancies": [],
            "confidence": "low",
            "checked_at": datetime.now().isoformat()
        }

def get_total_pages(pdf_path: str) -> int:
    """Get total number of pages in the PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)

def main():
    parser = argparse.ArgumentParser(description="Fact-check the final analysis against original book pages.")
    parser.add_argument("--pdf_path", default="$100m Offers.pdf", help="Path to the original PDF file")
    parser.add_argument("--analysis_document", default=DEFAULT_FINAL_DOCUMENT, help="Path to the final analysis document")
    parser.add_argument("--start_page", type=int, default=1, help="Starting page for fact-checking (1-indexed)")
    parser.add_argument("--end_page", type=int, default=None, help="Ending page for fact-checking (1-indexed)")
    parser.add_argument("--reset", action="store_true", help="Reset fact-check progress and start fresh")
    args = parser.parse_args()
    
    print("🔍 Starting Fact-Check Phase (Phase 7)")
    print("=" * 50)
    
    # Handle reset if requested
    if args.reset:
        if os.path.exists(FACT_CHECK_STATE_FILE):
            os.remove(FACT_CHECK_STATE_FILE)
            print("🗑️  Reset fact-check state")
        if os.path.exists(CONTROL_LOG_FILE):
            os.remove(CONTROL_LOG_FILE)
            print("🗑️  Reset control log")
    
    # Load final analysis document
    print(f"📄 Loading final analysis document: {args.analysis_document}")
    try:
        final_analysis = load_final_document(args.analysis_document)
        print(f"✅ Loaded analysis document ({len(final_analysis)} characters)")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure you have run the complete pipeline and have the final analysis document.")
        return
    
    # Initialize or load state
    state = initialize_fact_check_state()
    
    # Get total pages
    total_pages = get_total_pages(args.pdf_path)
    state["total_pages"] = total_pages
    
    # Determine page range
    start_page = max(args.start_page, state["current_page"])
    end_page = args.end_page or total_pages
    
    if start_page > state["current_page"] and not args.reset:
        print(f"📍 Resuming fact-check from page {start_page}")
    
    page_range = range(start_page, end_page + 1)
    
    if not page_range:
        print("✅ Fact-checking already complete!")
        return
    
    print(f"🔄 Fact-checking {len(page_range)} pages ({start_page} to {end_page})")
    print(f"📊 Progress: {state['completed_pages']}/{total_pages} pages completed")
    
    # Load existing control log
    control_log = state.get("control_log", [])
    
    failed_pages = []
    issues_found = 0
    
    with tqdm(total=len(page_range), desc="Fact-checking pages", unit="page") as pbar:
        for page_num in page_range:
            try:
                pbar.set_description(f"Fact-checking page {page_num}")
                
                # Extract page content
                page_data = extract_page_content(args.pdf_path, page_num)
                
                # Perform fact-check
                fact_check_result = fact_check_page_against_analysis(
                    page_num, 
                    page_data["text"], 
                    page_data["image_path"], 
                    final_analysis
                )
                
                # Add to control log
                control_log.append(fact_check_result)
                
                # Track issues
                if fact_check_result.get("issues_found", False):
                    issues_found += 1
                
                # Update state
                state["current_page"] = page_num + 1
                state["completed_pages"] += 1
                state["control_log"] = control_log
                save_fact_check_state(state)
                
                # Save control log periodically
                save_control_log(control_log)
                
                pbar.set_postfix({
                    'issues': issues_found,
                    'failed': len(failed_pages)
                })
                
            except Exception as e:
                print(f"\n❌ Critical error fact-checking page {page_num}: {str(e)}")
                failed_pages.append(page_num)
                
                # Add error entry to control log
                control_log.append({
                    "page_number": page_num,
                    "status": "critical_error",
                    "reason": str(e),
                    "issues_found": False,
                    "discrepancies": [],
                    "confidence": "low",
                    "checked_at": datetime.now().isoformat()
                })
                
            pbar.update(1)
    
    # Final save
    save_control_log(control_log)
    
    # Mark fact-checking as complete
    state["phase"] = "completed"
    state["completed_at"] = datetime.now().isoformat()
    save_fact_check_state(state)
    
    # Summary
    print("\n" + "="*50)
    print("✅ FACT-CHECK COMPLETE")
    print("="*50)
    print(f"📊 Total pages checked: {state['completed_pages']}")
    print(f"🚨 Pages with potential issues: {issues_found}")
    print(f"❌ Failed pages: {len(failed_pages)}")
    if failed_pages:
        print(f"   Failed page numbers: {failed_pages}")
    print(f"📋 Control log saved to: {CONTROL_LOG_FILE}")
    print(f"💾 State saved to: {FACT_CHECK_STATE_FILE}")
    
    if issues_found > 0:
        print(f"\n⚠️  Warning: {issues_found} pages have potential factual issues.")
        print("   Review the control log to assess the discrepancies.")
    else:
        print("\n🎉 No significant factual issues detected!")

if __name__ == "__main__":
    main() 