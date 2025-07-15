import argparse
import json
import os
import base64
import shutil
from pathlib import Path
from typing import Dict, Any, List

import pdfplumber
from pdf2image import convert_from_path
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
# Try to import the new Google Gen AI SDK, fallback to OpenAI if not available
try:
    from google import genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("⚠️  Google Gen AI SDK not found. Install with: pip install google-genai")
    print("   Falling back to OpenAI for overview processing.")

# Constants
STATE_FILE = "book_analysis_state.json"
MD_FILE = "book_analysis.md"
EXTRACTED_DIR = "extracted_pages"
SMALL_MODEL = "gpt-4.1-mini"  # Fallback for overview if Google not available
BIG_MODEL = "gpt-4.1"        # For detailed analysis and final synthesis
GEMINI_MODEL = "gemini-2.5-flash-lite-preview-06-17"  # Google's Flash-Lite preview model

# Section update handling constants
MAX_UPDATE_RETRIES = 3
FALLBACK_TO_FULL_DOC = True

load_dotenv()

# Initialize clients
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")  # Add this to your .env file

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in the .env file")

openai_client = OpenAI(api_key=openai_api_key)

# Initialize Google client if available
google_client = None
if GOOGLE_AVAILABLE and google_api_key:
    try:
        google_client = genai.Client(api_key=google_api_key)
        print("✅ Google Gen AI SDK initialized with Gemini 2.5 Flash-Lite")
    except Exception as e:
        print(f"⚠️  Failed to initialize Google client: {e}")
        print("   Falling back to OpenAI for overview processing.")
elif not google_api_key:
    print("⚠️  GOOGLE_API_KEY not found in .env file")
    print("   Add GOOGLE_API_KEY to use Gemini 2.5 Flash-Lite for faster overview processing.")

def extract_pdf_content(pdf_path: str, start_page: int, force_extract: bool = False) -> Dict[int, Dict[str, Any]]:
    """Extract text and images from PDF starting from a given page."""
    content = {}
    Path(EXTRACTED_DIR).mkdir(exist_ok=True)
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"PDF has {total_pages} pages. Checking for existing extractions...")
        
        # Check if pages are already extracted
        pages_to_extract = []
        for page_num in range(start_page - 1, total_pages):  # 0-indexed
            img_path = Path(EXTRACTED_DIR) / f"page_{page_num + 1}.png"
            if force_extract or not img_path.exists():
                pages_to_extract.append(page_num)
        
        if pages_to_extract:
            print(f"Extracting {len(pages_to_extract)} pages from PDF (starting from page {start_page})...")
            with tqdm(total=len(pages_to_extract), desc="Extracting PDF pages", unit="page") as extract_pbar:
                for page_num in pages_to_extract:
                    page = pdf.pages[page_num]
                    text = page.extract_text() or ""
                    
                    # Convert page to image
                    images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
                    img_path = Path(EXTRACTED_DIR) / f"page_{page_num + 1}.png"
                    images[0].save(img_path, "PNG")
                    extract_pbar.update(1)
        else:
            print("✅ All pages already extracted, skipping PDF extraction.")
        
        # Load all content (whether newly extracted or existing)
        for page_num in range(start_page - 1, total_pages):  # 0-indexed
            page = pdf.pages[page_num]
            text = page.extract_text() or ""
            img_path = Path(EXTRACTED_DIR) / f"page_{page_num + 1}.png"
            content[page_num + 1] = {"text": text, "image_path": str(img_path)}
    
    return content

def initialize_state() -> Dict[str, Any]:
    """Initialize or load the state JSON with lean structure."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "overview": "",                    # Single evolving string
        "current_phase": "overview",       # "overview" | "analysis" | "done"
        "current_page": 1,                 # Next page to process in current loop
        "prev_page_text": "",              # Previous page text for continuity
        "md_content": ""                   # Used only during analysis loop
    }

def save_state(state: Dict[str, Any]):
    """Save the state to JSON."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def update_overview(current_overview: str, page_num: int, page_text: str, img_path: str, prev_page_text: str = "") -> str:
    """Update overview using Gemini 2.5 Flash-Lite (fastest) or fallback to OpenAI small model."""
    try:
        # Skip if page has no meaningful content
        if not page_text or len(page_text.strip()) < 10:
            print(f"    ⚠️  Page {page_num} has minimal content, skipping overview update")
            return current_overview
        
        # Include previous page context for better inter-page understanding
        context_info = ""
        if prev_page_text:
            context_info = f"\nPrevious page context (for continuity): {prev_page_text}"
        
        prompt = f"""You are ONLY updating the book overview. DO NOT extract concepts, learnings, actions, or any specific content.

ONLY refine this overview with the new page's content. Focus ONLY on the book's central thesis, structure, and overall purpose.

Current Overview: {current_overview if current_overview else "None yet - create initial overview"}

{context_info}

Current page {page_num} text: {page_text}

RETURN ONLY THE UPDATED OVERVIEW TEXT. DO NOT format as JSON. DO NOT extract anything else."""

        # Try Google Gemini 2.5 Flash-Lite first (fastest and cheapest)
        if google_client:
            try:
                response = google_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[prompt]
                )
                
                if response and response.text:
                    return response.text.strip()
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
                {"role": "system", "content": "You ONLY update the book overview. DO NOT extract concepts, learnings, actions, or any other content. Return ONLY the updated overview text as plain text, not JSON."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ]
        )
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            print(f"    ⚠️  Empty response from OpenAI for page {page_num}, keeping previous overview")
            return current_overview
            
    except Exception as e:
        print(f"    ❌ Error updating overview for page {page_num}: {str(e)}")
        return current_overview

def create_base_document(overview: str):
    """Create a tailored base document using the big model based on the book overview."""
    
    prompt = f"""Based on this book overview, create a strategic business analysis document that will capture and organize the book's actual content and insights.

BOOK OVERVIEW:
{overview}

Create a comprehensive analysis document that:
1. Accurately captures the book's actual methodology and insights as they are presented
2. Preserves specific examples, personal anecdotes, and concrete details from the source
3. Distinguishes between the author's personal experiences and universal business principles
4. Provides space for actionable implementation steps based on what the book actually teaches
5. Maintains source fidelity while organizing content for practical business application

IMPORTANT GUIDELINES:
- Structure the document to accommodate the book's actual content rather than imposing external frameworks
- Create sections that can preserve specific examples and personal stories
- Avoid creating universal claims or frameworks not explicitly supported by the source
- Focus on being a faithful representation that business leaders can trust for accuracy

Return a markdown document structure that will be iteratively filled with the book's actual content."""

    try:
        response = openai_client.chat.completions.create(
            model=BIG_MODEL,
            messages=[
                {"role": "system", "content": "You are creating a document structure for business book analysis. Prioritize source fidelity and accuracy over polished frameworks. The goal is to faithfully capture what the book actually teaches, not to create idealized business frameworks."},
                {"role": "user", "content": prompt}
            ]
        )
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            base_document = response.choices[0].message.content
            with open(MD_FILE, "w") as f:
                f.write(base_document)
            print(f"📄 Created tailored base document ({len(base_document)} characters)")
        else:
            print("⚠️  Failed to create base document, using fallback")
            fallback_document = f"""# Business Book Analysis: Source-Faithful Summary

## Book Overview
{overview}

## Key Insights and Methodologies
*Specific insights and approaches as presented in the book*

## Personal Examples and Case Studies
*Author's personal experiences and specific examples*

## Actionable Business Principles
*Practical applications based on the book's actual teachings*

## Important Distinctions
*Differentiating between personal anecdotes and universal principles*

---
*This analysis prioritizes accuracy to the source material over framework creation.*
"""
            with open(MD_FILE, "w") as f:
                f.write(fallback_document)
                
    except Exception as e:
        print(f"❌ Error creating base document: {str(e)}")
        # Fallback to simple structure
        fallback_document = f"""# Business Book Analysis

## Book Overview
{overview}

## Core Insights
*Key insights as presented in the source material*

## Specific Examples
*Concrete examples and anecdotes from the book*

---
*Analysis focused on source fidelity*
"""
        with open(MD_FILE, "w") as f:
            f.write(fallback_document)

def parse_markdown_sections(document: str) -> Dict[str, str]:
    """Parse markdown document into sections based on headers."""
    sections = {}
    current_section = ""
    current_content = []
    
    lines = document.split('\n')
    
    for line in lines:
        # Check if this is a header line
        if line.strip().startswith('#'):
            # Save previous section if it exists
            if current_section:
                sections[current_section] = '\n'.join(current_content)
            
            # Start new section
            current_section = line.strip()
            current_content = [line]
        else:
            current_content.append(line)
    
    # Save the last section
    if current_section:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

def apply_section_updates(current_document: str, updates_text: str) -> tuple[str, bool]:
    """Apply section-based updates to the document.
    
    Updates format:
    SECTION: ## Header Name
    ACTION: REPLACE|APPEND|INSERT_AFTER:## Other Header
    CONTENT:
    [content here]
    ---
    """
    try:
        if updates_text.strip().upper() == "NO_CHANGES_NEEDED":
            return current_document, True
            
        # Parse current document into sections
        sections = parse_markdown_sections(current_document)
        
        # Parse updates
        update_blocks = updates_text.split('---')
        
        for block in update_blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
                
            # Parse update block
            section_line = None
            action_line = None
            content_start = None
            
            for i, line in enumerate(lines):
                if line.startswith('SECTION:'):
                    section_line = line[8:].strip()
                elif line.startswith('ACTION:'):
                    action_line = line[7:].strip()
                elif line.startswith('CONTENT:'):
                    content_start = i + 1
                    break
            
            if not all([section_line, action_line, content_start is not None]):
                print(f"    ❌ Malformed update block detected - triggering retry")
                return current_document, False
                
            # Get content
            content = '\n'.join(lines[content_start:]).strip()
            
            # Apply the update
            if action_line == "REPLACE":
                sections[section_line] = content
                print(f"    ✅ Replaced section: {section_line}")
                
            elif action_line == "APPEND":
                if section_line in sections:
                    sections[section_line] += "\n\n" + content
                else:
                    sections[section_line] = content
                print(f"    ✅ Appended to section: {section_line}")
                
            elif action_line.startswith("INSERT_AFTER:"):
                after_section = action_line[13:].strip()
                # This is more complex - for now, just add as new section
                sections[section_line] = content
                print(f"    ✅ Inserted new section: {section_line}")
            
            else:
                print(f"    ⚠️  Unknown action: {action_line}")
        
        # Rebuild document from sections
        # Try to maintain original order, add new sections at end
        doc_lines = current_document.split('\n')
        result_lines = []
        used_sections = set()
        
        i = 0
        while i < len(doc_lines):
            line = doc_lines[i]
            
            # Check if this starts a section we have updates for
            if line.strip().startswith('#'):
                section_header = line.strip()
                if section_header in sections:
                    # Use updated section content
                    result_lines.extend(sections[section_header].split('\n'))
                    used_sections.add(section_header)
                    
                    # Skip original section content
                    i += 1
                    while i < len(doc_lines) and not doc_lines[i].strip().startswith('#'):
                        i += 1
                    continue
            
            result_lines.append(line)
            i += 1
        
        # Add any new sections that weren't in the original document
        for section_header, content in sections.items():
            if section_header not in used_sections:
                result_lines.append('')
                result_lines.extend(content.split('\n'))
        
        updated_document = '\n'.join(result_lines)
        
        # Basic validation
        if len(updated_document.strip()) == 0:
            print(f"    ⚠️  Section updates resulted in empty document")
            return current_document, False
            
        return updated_document, True
        
    except Exception as e:
        print(f"    ❌ Section update error: {str(e)}")
        return current_document, False

def validate_section_updates(updates_content: str, page_text: str) -> tuple[bool, List[str]]:
    """Use small model to intelligently validate section updates for source fidelity.
    
    Returns:
        tuple: (is_valid, list_of_warnings)
    """
    
    validation_prompt = f"""You are a fact-checking specialist. Compare this page analysis update against the original source page to identify any accuracy issues.

ORIGINAL PAGE CONTENT:
{page_text}

PROPOSED UPDATE:
{updates_content}

Check for these specific issues:
1. OVERGENERALIZATION: Does the update make universal claims ("all businesses", "every company") when the source only discusses specific examples or the author's personal experience?
2. EXAMPLE SUBSTITUTION: Does the update use different examples than what appears in the source page?
3. MISSING SPECIFICS: Does the update lose specific details, personal anecdotes, or concrete examples from the source?
4. UNSUPPORTED CLAIMS: Does the update make claims about frameworks, statistics, or outcomes not found in this specific page?
5. CONTEXT LOSS: Does the update convert personal stories into generic business principles?

Respond with a JSON object:
{{
    "is_valid": true/false,
    "warnings": [
        "Specific issue description",
        "Another issue if found"
    ]
}}

If no issues found, return: {{"is_valid": true, "warnings": []}}"""

    try:
        response = openai_client.chat.completions.create(
            model=SMALL_MODEL,
            messages=[
                {"role": "system", "content": "You are a fact-checking specialist focused on source fidelity. Return only valid JSON with validation results."},
                {"role": "user", "content": validation_prompt}
            ]
        )
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            result_text = response.choices[0].message.content.strip()
            
            # Clean up any markdown formatting
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            elif result_text.startswith("```"):
                result_text = result_text[3:-3]
            
            try:
                result = json.loads(result_text)
                is_valid = result.get("is_valid", True)
                warnings = result.get("warnings", [])
                return is_valid, warnings
            except json.JSONDecodeError:
                print(f"    ⚠️  Invalid JSON from validation, assuming valid")
                return True, []
        else:
            return True, []  # Assume valid if no response
            
    except Exception as e:
        print(f"    ⚠️  Validation error: {str(e)}, assuming valid")
        return True, []

def patch_validation_issues(updates_content: str, page_text: str, warnings: List[str]) -> str:
    """Use small model to patch validation issues in the updates."""
    
    warnings_text = "\n".join([f"- {warning}" for warning in warnings])
    
    patch_prompt = f"""Fix the validation issues in this content update while preserving the original intent and structure.

ORIGINAL PAGE CONTENT:
{page_text}

CURRENT UPDATE (with issues):
{updates_content}

VALIDATION WARNINGS TO FIX:
{warnings_text}

INSTRUCTIONS:
- Fix only the specific validation issues mentioned
- Preserve all specific examples, personal anecdotes, and concrete details from the original page
- Remove overgeneralized claims not supported by the source
- Keep the exact same format (SECTION/ACTION/CONTENT structure)
- If personal experience is mentioned, keep it personal - don't make universal claims
- Use only information that appears in the original page content

Return the corrected update content with the same format, or return "NO_FIX_POSSIBLE" if the issues cannot be resolved."""

    try:
        response = openai_client.chat.completions.create(
            model=SMALL_MODEL,
            messages=[
                {"role": "system", "content": "You are a validation patch specialist. Fix only the specific issues mentioned while preserving the original content structure and source fidelity."},
                {"role": "user", "content": patch_prompt}
            ]
        )
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            patched_content = response.choices[0].message.content.strip()
            if patched_content == "NO_FIX_POSSIBLE":
                return updates_content  # Return original if no fix possible
            return patched_content
        else:
            return updates_content
            
    except Exception as e:
        print(f"    ❌ Patch error: {str(e)}")
        return updates_content

def generate_section_update_prompt(overview: str, page_num: int, page_text: str, current_document: str) -> str:
    """Generate the prompt for section-based document updates."""
    
    # Parse current sections for reference
    sections = parse_markdown_sections(current_document)
    section_list = "\n".join([f"- {header}" for header in sections.keys()])
    
    prompt = f"""You are updating a BUSINESS ACTION SHEET based on new page content. Use section-based updates for better reliability.

BOOK OVERVIEW:
{overview}

CURRENT DOCUMENT SECTIONS:
{section_list}

CURRENT DOCUMENT:
{current_document}

NEW PAGE {page_num} CONTENT:
{page_text}

CRITICAL INSTRUCTIONS FOR SOURCE FIDELITY:
1. PRESERVE SPECIFIC EXAMPLES: If the page contains specific examples, personal anecdotes, or concrete details, maintain them exactly as presented. Do not substitute with different examples.
2. DISTINGUISH PERSONAL vs UNIVERSAL: Clearly differentiate between the author's personal experiences and claims about universal business outcomes. Do not generalize personal anecdotes into universal business principles.
3. USE ONLY THIS PAGE'S CONTENT: Focus solely on what appears on this specific page. Do not blend in concepts or frameworks from other parts of the book.
4. MAINTAIN NARRATIVE CONTEXT: If the page contains personal stories or specific business scenarios, preserve the narrative context rather than converting to abstract frameworks.
5. AVOID OVERGENERALIZATION: Do not transform specific revenue figures, client results, or personal experiences into broad claims about "all businesses" or universal outcomes.

TASK:
1. Analyze the new page content for valuable business insights, frameworks, or actionable elements
2. Determine which sections (if any) need updates
3. Generate section-based updates using the format below
4. PRESERVE the authenticity and specificity of the source material

FORMAT:
Each update block MUST have exactly these 3 lines in this order:

SECTION: ## Section Header Name
ACTION: REPLACE
CONTENT:
[complete new section content including the header]
---
SECTION: ## Another Section  
ACTION: APPEND
CONTENT:
[content to add to existing section]
---

EXAMPLE:
SECTION: ## Key Insights and Methodologies
ACTION: REPLACE
CONTENT:
## Key Insights and Methodologies
- The author's personal experience with financial crisis
- Specific revenue growth from $100K to $1.2M/month
---

FORMAT REQUIREMENTS:
- Every block needs: SECTION: line, ACTION: line, CONTENT: line
- Use "---" to separate blocks
- Use exact section headers from the sections list above
- ACTION must be either "REPLACE" or "APPEND" (nothing else)

RULES:
- Use exact section headers from the current document sections list above
- ACTION can be: REPLACE (replace entire section) or APPEND (add to existing section)
- Include the full section content with proper markdown formatting
- If creating a new section, use REPLACE action
- If no changes needed, return "NO_CHANGES_NEEDED"
- Focus on practical business value while maintaining source accuracy
- Make targeted updates - don't rewrite everything
- PRESERVE specific examples, personal anecdotes, and concrete details exactly as they appear

Return your section updates or "NO_CHANGES_NEEDED"."""

    return prompt

def analyze_and_generate_section_updates(overview: str, page_num: int, page_text: str, img_path: str, current_document: str) -> tuple[str, bool]:
    """Generate section-based updates for document.
    
    Returns:
        tuple: (updates_content, is_updates_format)
    """
    
    # Skip if page has no meaningful content
    if not page_text or len(page_text.strip()) < 5:
        print(f"    ⚠️  Page {page_num} has minimal content, skipping")
        return "NO_CHANGES_NEEDED", True
    
    prompt = generate_section_update_prompt(overview, page_num, page_text, current_document)

    try:
        with open(img_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        
        response = openai_client.chat.completions.create(
            model=BIG_MODEL,
            messages=[
                {"role": "system", "content": "You are a business analysis specialist focused on source fidelity. Generate section-based updates that preserve specific examples, personal anecdotes, and concrete details from the source material. Distinguish between personal experiences and universal claims. Use the exact format requested and prioritize accuracy over polished frameworks."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ]
        )
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            updates_content = response.choices[0].message.content.strip()
            
            # Check if AI returned "no changes needed"
            if updates_content.upper().strip() == "NO_CHANGES_NEEDED":
                return "NO_CHANGES_NEEDED", True
                
            # Validate that it looks like section updates
            if "SECTION:" in updates_content and "ACTION:" in updates_content and "CONTENT:" in updates_content:
                return updates_content, True
            else:
                print(f"    ⚠️  AI returned invalid update format for page {page_num}")
                return updates_content, False
        else:
            print(f"    ⚠️  Empty response for page {page_num}")
            return current_document, False
            
    except Exception as e:
        print(f"    ❌ Error generating section updates for page {page_num}: {str(e)}")
        return current_document, False

def analyze_and_update_document(overview: str, page_num: int, page_text: str, img_path: str, current_document: str) -> str:
    """Analyze page content and return the updated document using diff-based approach with retry logic."""
    
    # Skip if page has no meaningful content
    if not page_text or len(page_text.strip()) < 5:
        print(f"    ⚠️  Page {page_num} has minimal content, skipping")
        return current_document
    
    # Try section-based approach with retries
    for attempt in range(MAX_UPDATE_RETRIES):
        try:
            # Generate section updates
            updates_content, is_updates_format = analyze_and_generate_section_updates(overview, page_num, page_text, img_path, current_document)
            
            # Handle "no changes needed" case
            if updates_content == "NO_CHANGES_NEEDED":
                print(f"    ✅ Page {page_num}: No changes needed")
                return current_document
            
            # If we got valid updates, try to apply them
            if is_updates_format:
                # Optional validation check for overgeneralization
                is_valid, warnings = validate_section_updates(updates_content, page_text)
                if warnings:
                    print(f"    ⚠️  Page {page_num}: Validation warnings detected:")
                    for warning in warnings:
                        print(f"       - {warning}")
                    
                    # Use small model to patch the issues
                    print(f"    🔧 Using small model to patch validation issues...")
                    patched_updates = patch_validation_issues(updates_content, page_text, warnings)
                    if patched_updates and patched_updates != updates_content:
                        print(f"    ✅ Validation patch applied")
                        updates_content = patched_updates
                    else:
                        print(f"    ⚠️  Validation patch failed, using original")
                
                updated_document, success = apply_section_updates(current_document, updates_content)
                
                if success:
                    print(f"    ✅ Page {page_num}: Section updates applied successfully (attempt {attempt + 1})")
                    return updated_document
                else:
                    print(f"    ⚠️  Page {page_num}: Section update application failed (attempt {attempt + 1}/{MAX_UPDATE_RETRIES})")
                    if attempt < MAX_UPDATE_RETRIES - 1:
                        print(f"       Retrying section update generation...")
                        continue
            else:
                print(f"    ⚠️  Page {page_num}: Invalid update format (attempt {attempt + 1}/{MAX_UPDATE_RETRIES})")
                if attempt < MAX_UPDATE_RETRIES - 1:
                    print(f"       Retrying section update generation...")
                    continue
                    
        except Exception as e:
            print(f"    ❌ Error in section update approach for page {page_num} (attempt {attempt + 1}/{MAX_UPDATE_RETRIES}): {str(e)}")
            if attempt < MAX_UPDATE_RETRIES - 1:
                print(f"       Retrying...")
                continue
    
    # Fallback to full document mode if all diff attempts failed
    if FALLBACK_TO_FULL_DOC:
        print(f"    🔄 Page {page_num}: Falling back to full document mode")
        return analyze_and_update_document_fullmode(overview, page_num, page_text, img_path, current_document)
    else:
        print(f"    ❌ Page {page_num}: All section update attempts failed, keeping current document")
        return current_document

def analyze_and_update_document_fullmode(overview: str, page_num: int, page_text: str, img_path: str, current_document: str) -> str:
    """Fallback function: Analyze page content and return the complete updated document (original approach)."""
    
    prompt = f"""You are updating a BUSINESS ACTION SHEET based on new page content. Return the COMPLETE updated document.

BOOK OVERVIEW:
{overview}

CURRENT ACTION SHEET:
{current_document}

NEW PAGE {page_num} CONTENT:
{page_text}

INSTRUCTIONS:
1. Read the current action sheet and the new page content
2. Identify any valuable business insights, frameworks, or actionable elements from the new page
3. Update the document ONLY where necessary - don't overdo it
4. Maintain the document's coherence and flow
5. Focus on practical business application and strategic value

IMPORTANT:
- Return the COMPLETE updated document (not just changes)
- Only make necessary updates - don't rewrite everything
- Keep the existing structure unless new content requires changes
- Focus on actionable business insights, not content summary
- Make this useful for entrepreneurs and business developers

Return the full updated markdown document."""

    try:
        with open(img_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        
        response = openai_client.chat.completions.create(
            model=BIG_MODEL,
            messages=[
                {"role": "system", "content": "You are a business strategist updating action sheets for entrepreneurs. Focus on practical value and strategic insights. Return complete updated documents, making only necessary changes."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ]
        )
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            print(f"    ✅ Page {page_num}: Full document mode successful")
            return response.choices[0].message.content.strip()
        else:
            print(f"    ⚠️  Empty response for page {page_num}, keeping current document")
            return current_document
            
    except Exception as e:
        print(f"    ❌ Error in full document mode for page {page_num}: {str(e)}")
        return current_document

def save_updated_document(updated_document: str) -> bool:
    """Save the complete updated document."""
    try:
        with open(MD_FILE, "w") as f:
            f.write(updated_document)
        return True
    except Exception as e:
        print(f"    ❌ Error saving document: {str(e)}")
        return False

def reset_analysis(force_reset: bool = False):
    """Reset analysis by removing state and extracted pages."""
    if force_reset:
        print("🗑️  Resetting analysis...")
        
        # Remove state file
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            print(f"   Removed {STATE_FILE}")
        
        # Remove markdown file
        if os.path.exists(MD_FILE):
            os.remove(MD_FILE)
            print(f"   Removed {MD_FILE}")
        
        # Remove extracted pages directory
        if os.path.exists(EXTRACTED_DIR):
            shutil.rmtree(EXTRACTED_DIR)
            print(f"   Removed {EXTRACTED_DIR}/")
        
        print("✅ Analysis reset complete.")

def main():
    parser = argparse.ArgumentParser(description="Analyze a book PDF with AI using incremental overview updates.")
    parser.add_argument("--pdf_path", default="$100m Offers.pdf", help="Path to the PDF file")
    parser.add_argument("--start_page", type=int, default=1, help="Starting page (1-indexed)")
    parser.add_argument("--reset", action="store_true", help="Force reset - delete all previous progress and start fresh")
    args = parser.parse_args()
    
    # Handle reset if requested
    if args.reset:
        reset_analysis(force_reset=True)
    
    # client = OpenAI(api_key=api_key) # This line is no longer needed as clients are initialized globally
    
    # Initialize or load state
    state = initialize_state()
    
    # Check current phase and resume accordingly
    if state["current_phase"] == "done":
        print("✅ Analysis already complete!")
        return
    elif state["current_phase"] == "overview" and state["current_page"] > 1 and not args.reset:
        print(f"📚 Resuming overview building from page {state['current_page']}")
        print(f"   Current progress: Overview length: {len(state['overview'])} characters")
    elif state["current_phase"] == "analysis" and state["current_page"] > 1 and not args.reset:
        print(f"📚 Resuming detailed analysis from page {state['current_page']}")
        print(f"   Overview complete: {len(state['overview'])} characters")
    
    # Extract content (with smart skip if already done)
    print("Extracting PDF content...")
    content = extract_pdf_content(args.pdf_path, args.start_page, force_extract=args.reset)
    
    # LOOP 1: Build Overview (if not done)
    if state["current_phase"] == "overview":
        print("\n🔄 LOOP 1: Building Overview...")
        
        # Determine actual starting page (resume from where we left off)
        actual_start_page = max(state["current_page"], args.start_page)
        if state["current_page"] > args.start_page and not args.reset:
            print(f"📍 Continuing from page {actual_start_page} (previous progress preserved)")
        
        page_range = range(actual_start_page, max(content.keys()) + 1)
        
        if not page_range:
            print("✅ Overview already complete!")
        else:
            print(f"🔄 Processing {len(page_range)} remaining pages for overview...")
            
            # Load previous page text for context if resuming
            prev_page_text = state.get("prev_page_text", "")
            if actual_start_page > 1 and (actual_start_page - 1) in content and not prev_page_text:
                prev_page_text = content[actual_start_page - 1]["text"]
                print(f"📖 Loaded previous page context for continuity")
            
            failed_pages = []
            
            with tqdm(total=len(page_range), desc="Building overview", unit="page") as pbar:
                for page_num in page_range:
                    try:
                        pbar.set_description(f"Overview page {page_num}")
                        page_data = content[page_num]
                        
                        # Update overview using small model (lightweight)
                        state["overview"] = update_overview(state["overview"], page_num, page_data["text"], page_data["image_path"], prev_page_text)
                        
                        # Update state
                        state["current_page"] = page_num + 1
                        state["prev_page_text"] = page_data["text"]
                        save_state(state)
                        
                        # Store current page text for next iteration's context
                        prev_page_text = page_data["text"]
                        
                        pbar.set_postfix({
                            'overview_chars': len(state['overview']), 
                            'failed': len(failed_pages)
                        })
                        
                    except Exception as e:
                        print(f"\n❌ Critical error processing page {page_num}: {str(e)}")
                        failed_pages.append(page_num)
                        # Continue with next page - don't break the entire process
                        
                    pbar.update(1)
            
            # Report any failed pages
            if failed_pages:
                print(f"\n⚠️  Warning: {len(failed_pages)} pages failed processing: {failed_pages}")
                print("Overview building continued with remaining pages.")
        
        # Overview complete - transition to analysis phase
        state["current_phase"] = "analysis"
        state["current_page"] = args.start_page  # Reset to start for analysis loop
        state["prev_page_text"] = ""  # Reset previous page text
        save_state(state)
        print(f"✅ Overview complete! ({len(state['overview'])} characters)")
    
    # LOOP 2: Detailed Analysis (if not done)
    if state["current_phase"] == "analysis":
        print("\n🔄 LOOP 2: Detailed Analysis...")
        
        # Create markdown skeleton if starting analysis
        if state["current_page"] == args.start_page:
            create_base_document(state["overview"])
            print("📄 Created markdown skeleton")
        
        # Determine actual starting page (resume from where we left off)
        actual_start_page = max(state["current_page"], args.start_page)
        if state["current_page"] > args.start_page:
            print(f"📍 Continuing analysis from page {actual_start_page}")
        
        page_range = range(actual_start_page, max(content.keys()) + 1)
        
        if not page_range:
            print("✅ Analysis already complete!")
        else:
            print(f"🔄 Processing {len(page_range)} remaining pages for analysis...")
            
            # Load previous page text for context if resuming
            prev_page_text = state.get("prev_page_text", "")
            if actual_start_page > 1 and (actual_start_page - 1) in content and not prev_page_text:
                prev_page_text = content[actual_start_page - 1]["text"]
                print(f"📖 Loaded previous page context for continuity")
            
            failed_pages = []
            
            with tqdm(total=len(page_range), desc="Analyzing pages", unit="page") as pbar:
                for page_num in page_range:
                    try:
                        pbar.set_description(f"Analyzing page {page_num}")
                        page_data = content[page_num]
                        
                        # Read current document
                        try:
                            with open(MD_FILE, "r") as f:
                                current_document = f.read()
                        except FileNotFoundError:
                            current_document = ""
                        
                        # Analyze page and get updated document
                        updated_document = analyze_and_update_document(state["overview"], page_num, page_data["text"], page_data["image_path"], current_document)
                        
                        # Save the updated document
                        save_updated_document(updated_document)
                        
                        # Update state
                        state["current_page"] = page_num + 1
                        state["prev_page_text"] = page_data["text"]
                        save_state(state)
                        
                        # Store current page text for next iteration's context
                        prev_page_text = page_data["text"]
                        
                        pbar.set_postfix({
                            'failed': len(failed_pages)
                        })
                        
                    except Exception as e:
                        print(f"\n❌ Critical error analyzing page {page_num}: {str(e)}")
                        failed_pages.append(page_num)
                        # Continue with next page - don't break the entire process
                        
                    pbar.update(1)
            
            # Report any failed pages
            if failed_pages:
                print(f"\n⚠️  Warning: {len(failed_pages)} pages failed processing: {failed_pages}")
                print("Analysis continued with remaining pages.")
        
        # Analysis complete
        state["current_phase"] = "done"
        # Clean up temporary state fields
        if "md_content" in state:
            del state["md_content"]
        save_state(state)
        print("✅ Analysis complete!")
    
    print(f"📊 Final state saved to {STATE_FILE}")
    print(f"📄 Final report saved to {MD_FILE}")
    print(f"📈 Overview length: {len(state['overview'])} characters")

if __name__ == "__main__":
    main() 