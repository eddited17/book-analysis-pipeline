import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import whatthepatch

# Constants
CONTROL_LOG_FILE = "fact_check_control_log.json"
FIX_STATE_FILE = "issue_fix_state.json"
FIX_REPORT_FILE = "issue_fix_report.json"
DEFAULT_FINAL_DOCUMENT = "book_analysis_final.md"
FIXED_DOCUMENT_SUFFIX = "_corrected.md"
BIG_MODEL = "gpt-4.1"  # Use GPT-4.1 for precise diff generation
SMALL_MODEL = "gpt-4.1-mini"  # Use GPT-4.1 for precise diff generation
EXTRACTED_DIR = "extracted_pages"

load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in the .env file")

openai_client = OpenAI(api_key=openai_api_key)

def load_control_log() -> Dict[str, Any]:
    """Load the fact-check control log."""
    try:
        with open(CONTROL_LOG_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Control log not found: {CONTROL_LOG_FILE}. Run fact-checking first.")

def load_document(document_path: str) -> str:
    """Load the document to be fixed."""
    try:
        with open(document_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Document not found: {document_path}")

def initialize_fix_state() -> Dict[str, Any]:
    """Initialize or load the fix state JSON."""
    if os.path.exists(FIX_STATE_FILE):
        with open(FIX_STATE_FILE, "r") as f:
            return json.load(f)
    else:
        return {
            "issues_processed": [],
            "issues_fixed": [],
            "issues_failed": [],
            "start_time": datetime.now().isoformat(),
            "total_character_changes": 0,
            "fix_summary": {}
        }

def save_fix_state(state: Dict[str, Any]):
    """Save the fix state JSON."""
    with open(FIX_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def extract_issues_from_control_log(control_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and prioritize issues from the control log."""
    issues = []
    
    # Get entries from the control log
    entries = control_log.get("entries", [])
    
    for entry in entries:
        if not entry.get("issues_found", False):
            continue
            
        page_num = entry.get("page_number")
        confidence = entry.get("confidence", "low")
        
        # Extract discrepancies
        for discrepancy in entry.get("discrepancies", []):
            issues.append({
                "type": "discrepancy",
                "page_number": page_num,
                "description": discrepancy,
                "confidence": confidence
            })
        
        # Extract missing content
        for missing in entry.get("missing_content", []):
            issues.append({
                "type": "missing_content",
                "page_number": page_num,
                "description": missing,
                "confidence": confidence
            })
        
        # Extract misrepresentations
        for misrep in entry.get("misrepresentations", []):
            issues.append({
                "type": "misrepresentation",
                "page_number": page_num,
                "description": misrep,
                "confidence": confidence
            })
    
    # Sort by priority: high confidence issues first, then by type importance
    priority_order = {
        "misrepresentation": 1,
        "missing_content": 2,
        "discrepancy": 3
    }
    
    confidence_order = {
        "high": 1,
        "medium": 2,
        "low": 3
    }
    
    issues.sort(key=lambda x: (
        confidence_order.get(x.get("confidence", "low"), 3),
        priority_order.get(x.get("type", "discrepancy"), 3)
    ))
    
    return issues

def get_priority(confidence: str, issue_type: str) -> int:
    """Get priority score for issue sorting."""
    confidence_weight = {"high": 10, "medium": 5, "low": 1}
    type_weight = {"misrepresentation": 3, "missing_content": 2, "discrepancy": 1}
    return confidence_weight.get(confidence, 1) * type_weight.get(issue_type, 1)

def get_page_content(page_num: int, pdf_path: str) -> str:
    """Get the original content from a specific page for reference."""
    # This is a placeholder - in practice you'd extract text from the PDF page
    # For now, return a note that this would contain the original page content
    return f"[Original content from page {page_num} would be extracted here]"

def find_relevant_section(document: str, issue: Dict[str, Any]) -> str:
    """Find the section of the document relevant to the issue."""
    description = issue.get('description', '').lower()
    page_num = issue.get('page_number', 0)
    
    # Look for keywords from the issue description in the document
    keywords = description.split()[:5]  # First 5 words
    lines = document.split('\n')
    
    # Find lines that contain any of the keywords
    relevant_lines = []
    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in keywords if len(keyword) > 3):
            # Include context: 10 lines before and after
            start = max(0, i - 10)
            end = min(len(lines), i + 11)
            relevant_lines.extend(list(range(start, end)))
    
    if relevant_lines:
        # Remove duplicates and sort
        relevant_lines = sorted(set(relevant_lines))
        return '\n'.join(lines[i] for i in relevant_lines)
    
    # Fallback: return section around page reference if found
    for i, line in enumerate(lines):
        if f"page {page_num}" in line.lower() or f"page{page_num}" in line.lower():
            start = max(0, i - 20)
            end = min(len(lines), i + 21)
            return '\n'.join(lines[start:end])
    
    # Last fallback: return first 3000 chars
    return document[:3000]

def generate_unified_diff(document: str, issue: Dict[str, Any], page_content: str, document_path: str) -> str:
    """
    Generate a standard unified diff using GPT-4.1 for precise, minimal changes.
    """
    
    # Find the relevant section
    relevant_section = find_relevant_section(document, issue)
    
    prompt = f"""Fix this specific issue by generating a unified diff.

ISSUE:
Type: {issue['type']}
Page: {issue['page_number']}
Description: {issue['description']}

RELEVANT DOCUMENT SECTION:
{relevant_section}

Generate a unified diff in this exact format:
--- a/{document_path}
+++ b/{document_path}
@@ -X,Y +X,Y @@
 context line
 context line
-old incorrect text
+new corrected text
 context line
 context line

 Return ONLY the unified diff, nothing else."""

    try:
        response = openai_client.chat.completions.create(
            model=BIG_MODEL,
            messages=[
                {"role": "system", "content": "You are a precision editor generating minimal unified diffs. Return only the unified diff format as specified."},
                {"role": "user", "content": prompt}
            ]
        )
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            diff_content = response.choices[0].message.content.strip()
            
            # Basic validation that it looks like a unified diff
            if "---" in diff_content and "+++" in diff_content and "@@" in diff_content:
                return diff_content
            else:
                print(f"    ⚠️  Invalid diff format generated")
                return ""
        else:
            print(f"    ⚠️  Empty response for diff generation")
            return ""
            
    except Exception as e:
        print(f"    ❌ Error generating diff: {str(e)}")
        return ""

def apply_unified_diff(document: str, diff_content: str) -> Tuple[str, bool, str]:
    """
    Apply a unified diff to the document using whatthepatch library.
    
    Returns:
        tuple: (updated_document, success, error_message)
    """
    try:
        # Parse the diff using whatthepatch
        patches = list(whatthepatch.parse_patch(diff_content))
        
        if not patches:
            return document, False, "No valid patches found in diff"
        
        # Apply the first patch
        patch = patches[0]
        lines = document.split('\n')
        
        # Apply using whatthepatch
        result = whatthepatch.apply_diff(patch, lines)
        
        if result is None:
            return document, False, "Failed to apply patch"
        
        # Convert back to string
        updated_document = '\n'.join(result)
        return updated_document, True, "Applied successfully"
        
    except Exception as e:
        return document, False, f"Error applying diff: {str(e)}"

def validate_diff_application(original_document: str, updated_document: str, issue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the diff was applied correctly and the issue was addressed.
    
    Returns:
        dict: Validation results with success status and metrics
    """
    
    # Calculate character-level changes
    original_chars = len(original_document)
    updated_chars = len(updated_document)
    char_difference = updated_chars - original_chars
    
    # Check if any changes were actually made
    if original_document == updated_document:
        return {
            "success": False,
            "reason": "No changes detected in document",
            "character_changes": 0,
            "change_percentage": 0.0
        }
    
    # Calculate percentage of document changed
    # Using a simple difflib-based approach for change calculation
    import difflib
    differ = difflib.SequenceMatcher(None, original_document, updated_document)
    similarity = differ.ratio()
    change_percentage = (1 - similarity) * 100
    
    # Validate that changes are minimal (less than 1% of document)
    if change_percentage > 1.0:
        return {
            "success": False,
            "reason": f"Changes too extensive: {change_percentage:.2f}% of document modified",
            "character_changes": abs(char_difference),
            "change_percentage": change_percentage
        }
    
    # Check if specific issue keywords appear to be addressed
    issue_keywords = issue.get("description", "").lower().split()[:5]  # First 5 words
    changes_relevant = False
    
    # Get the changed sections
    for tag, i1, i2, j1, j2 in differ.get_opcodes():
        if tag in ['replace', 'delete', 'insert']:
            changed_text = updated_document[j1:j2].lower()
            if any(keyword in changed_text for keyword in issue_keywords):
                changes_relevant = True
                break
    
    return {
        "success": True,
        "character_changes": abs(char_difference),
        "change_percentage": change_percentage,
        "changes_seem_relevant": changes_relevant,
        "validation_notes": f"Applied {abs(char_difference)} character changes ({change_percentage:.3f}% of document)"
    }

def save_document(document: str, filepath: str):
    """Save the updated document to file."""
    with open(filepath, "w") as f:
        f.write(document)

def generate_fix_report(state: Dict[str, Any], issues: List[Dict[str, Any]]):
    """Generate a comprehensive report of the fixing process."""
    report = {
        "fix_session": {
            "start_time": state["start_time"],
            "end_time": datetime.now().isoformat(),
            "total_issues_processed": len(state["issues_processed"]),
            "issues_successfully_fixed": len(state["issues_fixed"]),
            "issues_failed": len(state["issues_failed"]),
            "total_character_changes": state["total_character_changes"]
        },
        "issues_fixed": state["issues_fixed"],
        "issues_failed": state["issues_failed"],
        "summary": state["fix_summary"]
    }
    
    with open(FIX_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Fix analysis issues using targeted diff-based edits")
    parser.add_argument("--document", "-d", default=DEFAULT_FINAL_DOCUMENT,
                       help=f"Document to fix (default: {DEFAULT_FINAL_DOCUMENT})")
    parser.add_argument("--output", "-o", help="Output file path (default: adds _corrected suffix)")
    parser.add_argument("--reset", action="store_true", help="Reset fix state and start fresh")
    parser.add_argument("--max-issues", type=int, help="Maximum number of issues to process")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.document):
        print(f"❌ Document not found: {args.document}")
        return
    
    # Load control log
    try:
        control_log = load_control_log()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Reset state if requested
    if args.reset and os.path.exists(FIX_STATE_FILE):
        os.remove(FIX_STATE_FILE)
        print("🔄 Reset fix state")
    
    # Initialize state
    state = initialize_fix_state()
    
    # Load document
    document = load_document(args.document)
    original_document = document  # Keep original for comparison
    
    # Extract and prioritize issues
    issues = extract_issues_from_control_log(control_log)
    
    # Filter out already processed issues
    remaining_issues = [issue for issue in issues 
                       if issue not in state["issues_processed"]]
    
    if args.max_issues:
        remaining_issues = remaining_issues[:args.max_issues]
    
    print(f"📋 Found {len(remaining_issues)} issues to process")
    
    if not remaining_issues:
        print("✅ No remaining issues to fix")
        return
    
    # Process each issue
    total_character_changes = state["total_character_changes"]
    
    for i, issue in enumerate(tqdm(remaining_issues, desc="Fixing issues")):
        print(f"\n🔧 Processing issue {i+1}/{len(remaining_issues)}")
        print(f"   Type: {issue['type']}")
        print(f"   Page: {issue['page_number']}")
        print(f"   Confidence: {issue['confidence']}")
        print(f"   Description: {issue['description'][:100]}...")
        
        # Get original page content for reference
        page_content = get_page_content(issue['page_number'], '$100m Offers.pdf')
        
        # Generate unified diff
        print("   🔍 Generating diff...")
        diff_content = generate_unified_diff(document, issue, page_content, args.document)
        
        if not diff_content:
            print("   ❌ Failed to generate diff")
            state["issues_failed"].append({
                "issue": issue,
                "reason": "Diff generation failed",
                "timestamp": datetime.now().isoformat()
            })
            continue
        
        # Apply the diff
        print("   ⚙️  Applying changes...")
        updated_document, success, error_msg = apply_unified_diff(document, diff_content)
        
        if not success:
            print(f"   ❌ Failed to apply diff: {error_msg}")
            state["issues_failed"].append({
                "issue": issue,
                "reason": f"Diff application failed: {error_msg}",
                "timestamp": datetime.now().isoformat()
            })
            continue
        
        # Validate the changes
        print("   ✅ Validating changes...")
        validation = validate_diff_application(document, updated_document, issue)
        
        if not validation["success"]:
            print(f"   ❌ Validation failed: {validation['reason']}")
            state["issues_failed"].append({
                "issue": issue,
                "reason": f"Validation failed: {validation['reason']}",
                "timestamp": datetime.now().isoformat()
            })
            continue
        
        # Success - update document and state
        document = updated_document
        char_changes = validation["character_changes"]
        total_character_changes += char_changes
        
        print(f"   ✅ Fixed! ({char_changes} characters changed)")
        
        state["issues_fixed"].append({
            "issue": issue,
            "character_changes": char_changes,
            "change_percentage": validation["change_percentage"],
            "timestamp": datetime.now().isoformat()
        })
        
        state["issues_processed"].append(issue)
        state["total_character_changes"] = total_character_changes
        
        # Save state after each successful fix
        save_fix_state(state)
    
    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        base_path = Path(args.document)
        output_path = base_path.parent / f"{base_path.stem}{FIXED_DOCUMENT_SUFFIX}"
    
    # Save the corrected document
    save_document(document, str(output_path))
    
    # Update final summary
    state["fix_summary"] = {
        "total_issues_found": len(issues),
        "issues_successfully_fixed": len(state["issues_fixed"]),
        "issues_failed": len(state["issues_failed"]),
        "total_character_changes": state["total_character_changes"],
        "original_document_size": len(original_document),
        "final_document_size": len(document),
        "percentage_changed": (state["total_character_changes"] / len(original_document)) * 100,
        "output_file": str(output_path)
    }
    
    # Save final state and generate report
    save_fix_state(state)
    generate_fix_report(state, issues)
    
    print(f"\n✅ Issue fixing complete!")
    print(f"📄 Corrected document saved to: {output_path}")
    print(f"📊 Fixed {len(state['issues_fixed'])} issues with {state['total_character_changes']} character changes")
    print(f"📈 Document changed by {state['fix_summary']['percentage_changed']:.3f}%")
    print(f"📋 Detailed report saved to: {FIX_REPORT_FILE}")

if __name__ == "__main__":
    main() 