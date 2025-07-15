# Book Analysis Pipeline

A sophisticated multi-stage AI-powered pipeline for analyzing and processing business books, specifically designed to extract actionable insights, frameworks, and implementation strategies from PDF documents.

## Overview

This pipeline transforms a PDF business book into a comprehensive, consolidated analysis document with zero content loss but optimized organization. It uses multiple AI models (OpenAI GPT-4.1, GPT-4.1-mini, Gemini 2.5 Flash-Lite, and O3) for different processing stages to maximize quality and efficiency.

## Pipeline Workflow

The pipeline consists of 7 distinct phases executed across 7 Python scripts:

### Phase 1: Document Extraction & Initial Analysis
**Script:** `analyze_book.py`
- Extracts text and images from PDF pages
- Creates incremental overview using Gemini 2.5 Flash-Lite (fastest) or GPT-4.1-mini
- Performs detailed page-by-page analysis using GPT-4.1
- Generates initial markdown document with actionable business insights
- **Output:** `book_analysis.md`, `book_analysis_state.json`, `extracted_pages/`

### Phase 2: Semantic Content Analysis
**Script:** `analyze_semantic_content.py`
- Identifies semantic boundaries and content chunks
- Creates content fingerprints for each chunk (frameworks, examples, action items)
- **Output:** `semantic_analysis_results.json`

### Phase 3: Similarity Analysis
**Script:** `run_similarity_analysis.py` 
- Analyzes semantic similarity between all content chunks
- Categorizes similarities as high (>80%), medium (50-80%), or low (<50%)
- **Output:** Updated `semantic_analysis_results.json` with similarity data

### Phase 4: Consolidation Mapping
**Script:** `create_consolidation_map.py`
- Creates intelligent consolidation strategy based on similarity analysis
- Defines merge rules for high similarity content
- Groups medium similarity content while preserving unique elements
- Preserves low similarity content as standalone sections
- **Output:** `consolidation_map.json`

### Phase 5: Document Consolidation
**Script:** `consolidate_document.py`
- Executes the consolidation plan using GPT-4.1
- Merges similar content while preserving all unique insights
- Creates professionally structured consolidated document
- **Output:** `book_analysis_consolidated.md`, `consolidation_report.json`

### Phase 6: Final Enhancement
**Script:** `finalize_with_o3.py`
- Uses OpenAI's O3 model for final presentation polish
- Improves formatting, creates clean table of contents, fixes headers
- Maintains 100% content preservation while enhancing readability
- **Output:** `book_analysis_final.md`, `finalization_report.json`

### Phase 7: Fact-Check Validation
**Script:** `fact_check_analysis.py`
- Validates final analysis against original book pages for factual accuracy
- Uses Gemini 2.5 Flash-Lite for efficient fact-checking (with OpenAI fallback)
- Identifies misrepresentations, omissions, and factual errors
- **Output:** `fact_check_control_log.json`, `fact_check_state.json`

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key with access to GPT-4.1, GPT-4.1-mini, and O3 models
- Google API key (optional, for faster overview processing with Gemini)

### Installation

1. **Clone or download the pipeline files**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file with:
```
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here  # Optional but recommended
```

## Usage

### Complete Pipeline Execution

**For a new book analysis:**
```bash
# Phase 1: Extract and analyze the book
python analyze_book.py path/to/your/book.pdf

# Phase 2: Semantic content analysis  
python analyze_semantic_content.py book_analysis.md

# Phase 3: Similarity analysis
python run_similarity_analysis.py

# Phase 4: Create consolidation mapping
python create_consolidation_map.py

# Phase 5: Consolidate document
python consolidate_document.py

# Phase 6: Final enhancement with O3
python finalize_with_o3.py

# Phase 7: Fact-check validation (only for internal testing. Still needs a lot of development)
python fact_check_analysis.py
```

### Advanced Usage Options

**Resume interrupted analysis:**
```bash
# The pipeline automatically resumes from the last processed page
python analyze_book.py path/to/book.pdf
```

**Force re-extraction of pages:**
```bash
python analyze_book.py path/to/book.pdf --start-page 1 --force-extract
```

**Custom output filenames:**
```bash
python finalize_with_o3.py book_analysis_consolidated.md my_final_output.md
```

**Fact-check specific page ranges:**
```bash
# Check only pages 50-100
python fact_check_analysis.py --start_page 50 --end_page 100

# Resume interrupted fact-checking
python fact_check_analysis.py

# Reset and start fresh
python fact_check_analysis.py --reset
```

## Output Files

### Primary Outputs
- **`book_analysis_final.md`** - The final polished business analysis document
- **`book_analysis_consolidated.md`** - Consolidated document before final polish
- **`book_analysis.md`** - Initial analysis document from phase 1

### Process Files
- **`semantic_analysis_results.json`** - Semantic chunks and similarity data
- **`consolidation_map.json`** - Consolidation strategy and rules
- **`book_analysis_state.json`** - Analysis progress state (for resuming)
- **`fact_check_state.json`** - Fact-checking progress state (for resuming)

### Reports & Logs
- **`finalization_report.json`** - Final enhancement process details
- **`consolidation_report.json`** - Consolidation process summary
- **`fact_check_control_log.json`** - Fact-checking results and discrepancy tracking
- **`preservation_log.txt`** - Content preservation verification
- **`completion_log.txt`** - Overall pipeline execution log

### Assets
- **`extracted_pages/`** - Individual page images from PDF

## Model Usage Strategy

The pipeline strategically uses different AI models for optimal cost and quality:

- **Gemini 2.5 Flash-Lite:** Overview updates (fastest, cheapest)
- **GPT-4.1-mini:** Semantic analysis and similarity detection (cost-effective)
- **GPT-4.1:** Detailed page analysis and consolidation (high quality)
- **O3:** Final enhancement and reasoning (superior polish)

## Key Features

### Content Preservation
- **Zero Content Loss:** All unique insights, frameworks, and examples are preserved
- **Similarity Detection:** Intelligently identifies redundant content for consolidation
- **Unique Element Tracking:** Explicitly preserves distinctive value from each section

### Professional Output
- **Actionable Focus:** Transforms book content into business implementation tools
- **Clean Formatting:** Professional markdown with clickable table of contents
- **Logical Structure:** Optimized flow and organization for reference use

### Resumable Processing
- **State Management:** Automatically saves progress and resumes from interruptions
- **Incremental Updates:** Only processes new or changed content
- **Error Recovery:** Graceful fallbacks for API failures

## Troubleshooting

### Common Issues

**"OPENAI_API_KEY must be set"**
- Ensure your `.env` file contains a valid OpenAI API key
- Verify the key has access to required models (GPT-4.1, O3)

**"Required file not found"**
- Run the previous phase scripts in order
- Check that output files from previous phases exist

**"Empty response from API"**
- Check your API quotas and rate limits
- Verify API keys are valid and have sufficient credits

**PDF extraction fails**
- Ensure the PDF is not password protected
- Check that pdf2image dependencies are properly installed
- For Linux: `sudo apt-get install poppler-utils`

### Performance Tips

1. **Use Google API key** for faster overview processing
2. **Run on a machine with good internet** for faster API calls
3. **Monitor API usage** as the pipeline makes many calls
4. **Start with smaller sections** to test the pipeline

## Cost Considerations

This pipeline makes extensive use of AI APIs. Estimated costs for a 200-page business book:
- **Phase 1:** $15-25 (detailed analysis)
- **Phases 2-4:** $3-5 (semantic processing)
- **Phase 5:** $5-10 (consolidation)
- **Phase 6:** $10-20 (O3 enhancement)
- **Phase 7:** $2-5 (fact-checking with Gemini 2.5 Flash-Lite)

**Total estimated cost:** $37-65 per book

## Customization

The pipeline can be customized by modifying:
- **Model selections** in each script's constants
- **Prompts** for different analysis focuses
- **Similarity thresholds** in similarity analysis
- **Consolidation strategies** in mapping logic

## Support

For issues or questions about the pipeline:
1. Check the completion logs for detailed error information
2. Verify all dependencies are correctly installed
3. Ensure API keys have sufficient credits and model access
4. Review the troubleshooting section above

## License

This pipeline is designed for business book analysis and educational purposes. Ensure you have proper rights to analyze any copyrighted materials. 