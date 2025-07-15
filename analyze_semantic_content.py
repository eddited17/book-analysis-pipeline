import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import openai
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ContentChunk:
    """Represents a semantic chunk of content with metadata."""
    id: str
    content: str
    start_line: int
    end_line: int
    topic_theme: str = ""
    frameworks: List[str] = None
    examples: List[str] = None
    action_items: List[str] = None
    unique_insights: List[str] = None
    
    def __post_init__(self):
        if self.frameworks is None:
            self.frameworks = []
        if self.examples is None:
            self.examples = []
        if self.action_items is None:
            self.action_items = []
        if self.unique_insights is None:
            self.unique_insights = []

@dataclass
class SimilarityAnalysis:
    """Represents similarity analysis between two chunks."""
    chunk1_id: str
    chunk2_id: str
    similarity_score: float
    similarity_category: str  # "high", "medium", "low"
    overlap_percentage: float
    unique_value_chunk1: str
    unique_value_chunk2: str
    overlap_description: str

class SemanticAnalyzer:
    def __init__(self, model_name: str = "gpt-4.1-mini"):  # GPT-4.1 Mini as specified in consolidation plan
        self.model_name = model_name
        # Initialize OpenAI client with environment variables
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")
        self.client = openai.OpenAI(api_key=api_key)
        
    def load_document(self, file_path: str) -> List[Dict]:
        """Load document and split into paragraphs."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into paragraphs, preserving line numbers
            lines = content.split('\n')
            paragraphs = []
            current_paragraph = []
            current_start = 0
            
            for i, line in enumerate(lines):
                if line.strip() == '':
                    if current_paragraph:
                        paragraphs.append({
                            'content': '\n'.join(current_paragraph),
                            'start_line': current_start + 1,
                            'end_line': i
                        })
                        current_paragraph = []
                else:
                    if not current_paragraph:
                        current_start = i
                    current_paragraph.append(line)
            
            # Add final paragraph if exists
            if current_paragraph:
                paragraphs.append({
                    'content': '\n'.join(current_paragraph),
                    'start_line': current_start + 1,
                    'end_line': len(lines)
                })
            
            return paragraphs
            
        except Exception as e:
            print(f"Error loading document: {e}")
            return []
    
    def identify_semantic_boundaries(self, paragraphs: List[Dict]) -> List[ContentChunk]:
        """Use GPT-4.1 mini to identify semantic boundaries and create chunks."""
        
        # Prepare content for analysis
        content_for_analysis = []
        for i, para in enumerate(paragraphs):
            content_for_analysis.append(f"[PARAGRAPH {i}] (Lines {para['start_line']}-{para['end_line']})\n{para['content']}")
        
        full_content = '\n\n'.join(content_for_analysis)
        
        prompt = f"""
        You are analyzing a business document to identify semantic boundaries and create meaningful content chunks.
        
        IGNORE existing headers and structural markers - they are unreliable from incremental analysis.
        
        Your task:
        1. Identify where distinct concepts/topics actually begin and end based on content meaning
        2. Group related paragraphs regardless of their structural placement
        3. Focus on semantic meaning, not structural markers
        4. Look for: frameworks, examples, action items, tools, principles
        
        Document content:
        {full_content}
        
        Please identify semantic chunks and return them in this exact JSON format:
        {{
            "chunks": [
                {{
                    "chunk_id": "chunk_1",
                    "paragraph_range": [start_paragraph_index, end_paragraph_index],
                    "main_concept": "Brief description of what this chunk is fundamentally about",
                    "content_type": "framework|example|action_items|tools|principles|overview"
                }}
            ]
        }}
        
        Focus on creating chunks that represent complete, coherent ideas rather than arbitrary text segments.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            print(f"Raw GPT response: {response_text[:500]}...")  # Debug output
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                print(f"Extracted JSON: {json_text[:300]}...")  # Debug output
                chunk_data = json.loads(json_text)
                print(f"Found {len(chunk_data.get('chunks', []))} chunks in response")  # Debug output
                
                # Create ContentChunk objects
                chunks = []
                for i, chunk_info in enumerate(chunk_data['chunks']):
                    paragraph_range = chunk_info['paragraph_range']
                    start_para = paragraph_range[0]
                    # Handle both single-element [start] and two-element [start, end] ranges
                    end_para = paragraph_range[1] if len(paragraph_range) > 1 else paragraph_range[0]
                    
                    # Validate paragraph indices
                    if start_para < 0 or start_para >= len(paragraphs):
                        print(f"Warning: start_para {start_para} out of range for {len(paragraphs)} paragraphs, skipping chunk {chunk_info.get('chunk_id', 'unknown')}")
                        continue
                    if end_para < 0 or end_para >= len(paragraphs):
                        print(f"Warning: end_para {end_para} out of range for {len(paragraphs)} paragraphs, adjusting to {len(paragraphs)-1}")
                        end_para = len(paragraphs) - 1
                    
                    # Ensure start_para <= end_para
                    if start_para > end_para:
                        print(f"Warning: start_para {start_para} > end_para {end_para}, swapping")
                        start_para, end_para = end_para, start_para
                    
                    # Combine paragraphs in range
                    chunk_content = []
                    start_line = paragraphs[start_para]['start_line']
                    end_line = paragraphs[end_para]['end_line']
                    
                    for para_idx in range(start_para, end_para + 1):
                        if para_idx < len(paragraphs):
                            chunk_content.append(paragraphs[para_idx]['content'])
                    
                    # Only create chunk if we have content
                    if chunk_content:
                        chunk = ContentChunk(
                            id=chunk_info['chunk_id'],
                            content='\n\n'.join(chunk_content),
                            start_line=start_line,
                            end_line=end_line,
                            topic_theme=chunk_info['main_concept']
                        )
                        chunks.append(chunk)
                
                return chunks
            else:
                print("No JSON found in GPT response")
                print(f"Full response: {response_text}")
                return []
                
        except Exception as e:
            print(f"Error in semantic boundary identification: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def create_content_fingerprint(self, chunk: ContentChunk) -> Dict[str, Any]:
        """Extract key concepts from each chunk to create a content signature."""
        
        prompt = f"""
        Analyze this content chunk and extract key concepts to create a "content fingerprint":
        
        Content:
        {chunk.content}
        
        Extract and identify:
        1. Main topic/theme: What is this chunk fundamentally about?
        2. Frameworks mentioned: Specific business frameworks or methodologies
        3. Examples used: Case studies, scenarios, or practical demonstrations  
        4. Action items provided: Concrete steps or tools mentioned
        5. Unique insights: What makes this chunk distinct from others?
        
        Return in this JSON format:
        {{
            "main_topic": "Brief description",
            "frameworks": ["framework1", "framework2"],
            "examples": ["example1", "example2"],
            "action_items": ["action1", "action2"],
            "unique_insights": ["insight1", "insight2"],
            "key_concepts": ["concept1", "concept2"],
            "content_category": "framework|example|implementation|tools|principles"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                fingerprint = json.loads(json_match.group())
                
                # Update chunk with extracted information
                chunk.topic_theme = fingerprint.get('main_topic', chunk.topic_theme)
                chunk.frameworks = fingerprint.get('frameworks', [])
                chunk.examples = fingerprint.get('examples', [])
                chunk.action_items = fingerprint.get('action_items', [])
                chunk.unique_insights = fingerprint.get('unique_insights', [])
                
                return fingerprint
                
        except Exception as e:
            print(f"Error creating content fingerprint for chunk {chunk.id}: {e}")
            return {}
    
    def analyze_similarity_between_chunks(self, chunk1: ContentChunk, chunk2: ContentChunk) -> SimilarityAnalysis:
        """Analyze semantic similarity between two chunks using GPT-4.1 Mini."""
        prompt = f"""
        You are analyzing semantic similarity between two business content chunks. Please provide detailed analysis.

        CHUNK 1:
        ID: {chunk1.id}
        Topic: {chunk1.topic_theme}
        Frameworks: {', '.join(chunk1.frameworks) if chunk1.frameworks else 'None'}
        Examples: {', '.join(chunk1.examples[:2]) if chunk1.examples else 'None'}  # Limit to first 2 examples
        Action Items: {', '.join(chunk1.action_items[:3]) if chunk1.action_items else 'None'}  # Limit to first 3
        Unique Insights: {', '.join(chunk1.unique_insights[:2]) if chunk1.unique_insights else 'None'}  # Limit to first 2
        
        CHUNK 2:
        ID: {chunk2.id}
        Topic: {chunk2.topic_theme}
        Frameworks: {', '.join(chunk2.frameworks) if chunk2.frameworks else 'None'}
        Examples: {', '.join(chunk2.examples[:2]) if chunk2.examples else 'None'}
        Action Items: {', '.join(chunk2.action_items[:3]) if chunk2.action_items else 'None'}
        Unique Insights: {', '.join(chunk2.unique_insights[:2]) if chunk2.unique_insights else 'None'}

        Please analyze these chunks and provide:
        1. Similarity score (0-100)
        2. Similarity category: "high" (>80%), "medium" (50-80%), or "low" (<50%)
        3. Overlap percentage (0-100) - what percentage of information overlaps
        4. Unique value of chunk 1 (what does it provide that chunk 2 doesn't)
        5. Unique value of chunk 2 (what does it provide that chunk 1 doesn't)
        6. Description of what overlaps between them

        Format your response as JSON:
        {{
            "similarity_score": <number>,
            "similarity_category": "<category>",
            "overlap_percentage": <number>,
            "unique_value_chunk1": "<description>",
            "unique_value_chunk2": "<description>",
            "overlap_description": "<description>"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert content analyst specializing in semantic similarity analysis for business content consolidation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                similarity_data = json.loads(json_match.group())
                
                return SimilarityAnalysis(
                    chunk1_id=chunk1.id,
                    chunk2_id=chunk2.id,
                    similarity_score=similarity_data.get("similarity_score", 0),
                    similarity_category=similarity_data.get("similarity_category", "low"),
                    overlap_percentage=similarity_data.get("overlap_percentage", 0),
                    unique_value_chunk1=similarity_data.get("unique_value_chunk1", ""),
                    unique_value_chunk2=similarity_data.get("unique_value_chunk2", ""),
                    overlap_description=similarity_data.get("overlap_description", "")
                )
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Error analyzing similarity between {chunk1.id} and {chunk2.id}: {e}")
            return SimilarityAnalysis(
                chunk1_id=chunk1.id,
                chunk2_id=chunk2.id,
                similarity_score=0,
                similarity_category="low",
                overlap_percentage=0,
                unique_value_chunk1="Analysis failed",
                unique_value_chunk2="Analysis failed",
                overlap_description="Analysis failed"
            )
    
    def analyze_all_similarities(self, chunks: List[ContentChunk]) -> List[SimilarityAnalysis]:
        """Analyze similarity between all pairs of chunks."""
        similarities = []
        total_pairs = len(chunks) * (len(chunks) - 1) // 2
        current_pair = 0
        
        print(f"Analyzing similarities between {total_pairs} chunk pairs...")
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                current_pair += 1
                print(f"Analyzing similarity {current_pair}/{total_pairs}: {chunks[i].id} vs {chunks[j].id}")
                
                similarity = self.analyze_similarity_between_chunks(chunks[i], chunks[j])
                similarities.append(similarity)
                
                # Print progress for high and medium similarity findings
                if similarity.similarity_category in ["high", "medium"]:
                    print(f"  → {similarity.similarity_category.upper()} similarity ({similarity.similarity_score}%): {similarity.overlap_description[:100]}...")
        
        return similarities
    
    def categorize_similarities(self, similarities: List[SimilarityAnalysis]) -> Dict[str, List[SimilarityAnalysis]]:
        """Categorize similarities by level."""
        categorized = {
            "high": [],
            "medium": [],
            "low": []
        }
        
        for sim in similarities:
            categorized[sim.similarity_category].append(sim)
        
        # Sort each category by similarity score (descending)
        for category in categorized:
            categorized[category].sort(key=lambda x: x.similarity_score, reverse=True)
        
        return categorized

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Main analysis function that processes the entire document."""
        
        print("Loading document...")
        paragraphs = self.load_document(file_path)
        print(f"Loaded {len(paragraphs)} paragraphs")
        
        print("Identifying semantic boundaries...")
        chunks = self.identify_semantic_boundaries(paragraphs)
        print(f"Created {len(chunks)} semantic chunks")
        
        print("Creating content fingerprints...")
        fingerprints = {}
        for chunk in chunks:
            fingerprint = self.create_content_fingerprint(chunk)
            fingerprints[chunk.id] = fingerprint
        
        print("Analyzing similarities between chunks...")
        similarities = self.analyze_all_similarities(chunks)
        print(f"Analyzed {len(similarities)} chunk pairs for similarity.")

        print("Analysis complete!")
        
        return {
            'chunks': chunks,
            'fingerprints': fingerprints,
            'similarities': similarities,
            'total_paragraphs': len(paragraphs),
            'total_chunks': len(chunks)
        }
    
    def save_analysis(self, analysis_result: Dict[str, Any], output_file: str):
        """Save analysis results to JSON file."""
        
        # Convert ContentChunk objects to dictionaries for JSON serialization
        chunks_dict = []
        for chunk in analysis_result['chunks']:
            chunk_dict = {
                'id': chunk.id,
                'content': chunk.content,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
                'topic_theme': chunk.topic_theme,
                'frameworks': chunk.frameworks,
                'examples': chunk.examples,
                'action_items': chunk.action_items,
                'unique_insights': chunk.unique_insights
            }
            chunks_dict.append(chunk_dict)
        
        # Convert SimilarityAnalysis objects to dictionaries for JSON serialization
        similarities_dict = []
        for sim in analysis_result['similarities']:
            sim_dict = {
                'chunk1_id': sim.chunk1_id,
                'chunk2_id': sim.chunk2_id,
                'similarity_score': sim.similarity_score,
                'similarity_category': sim.similarity_category,
                'overlap_percentage': sim.overlap_percentage,
                'unique_value_chunk1': sim.unique_value_chunk1,
                'unique_value_chunk2': sim.unique_value_chunk2,
                'overlap_description': sim.overlap_description
            }
            similarities_dict.append(sim_dict)

        output_data = {
            'chunks': chunks_dict,
            'fingerprints': analysis_result['fingerprints'],
            'similarities': similarities_dict,
            'total_paragraphs': analysis_result['total_paragraphs'],
            'total_chunks': analysis_result['total_chunks'],
            'analysis_metadata': {
                'model_used': self.model_name,
                'timestamp': str(Path(output_file).stat().st_mtime if Path(output_file).exists() else 'new')
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis saved to {output_file}")

def main():
    """Main execution function."""
    
    # Configuration
    input_file = "book_analysis.md"
    output_file = "semantic_analysis_results.json"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # Initialize analyzer
    analyzer = SemanticAnalyzer()
    
    # Perform analysis
    print("Starting semantic analysis...")
    try:
        results = analyzer.analyze_document(input_file)
        
        # Save results
        analyzer.save_analysis(results, output_file)
        
        # Print summary
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Total paragraphs processed: {results['total_paragraphs']}")
        print(f"Semantic chunks created: {results['total_chunks']}")
        print(f"Similarity analyses performed: {len(results['similarities'])}")
        print(f"Results saved to: {output_file}")
        
        # Display chunk overview
        print(f"\n=== CHUNK OVERVIEW ===")
        for chunk in results['chunks']:
            print(f"• {chunk.id}: {chunk.topic_theme} (Lines {chunk.start_line}-{chunk.end_line})")
            
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 