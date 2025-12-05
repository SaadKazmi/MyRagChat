# ---- Imports ----
import streamlit as st
import fitz  # PyMuPDF
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import google.generativeai as genai
import requests
import urllib.request
import xml.etree.ElementTree as ET
import gzip
import shutil
import tarfile
import time
import tempfile
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import uuid  # Add this import

# ---- Config ----
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "testingchat"
GEN_API_KEY = st.secrets["GEN_API_KEY"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
genai.configure(api_key=GEN_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
embeddings = OpenAIEmbeddings()

# ---- Session State Initialization ----
if "retrieved_papers" not in st.session_state:
    st.session_state.retrieved_papers = {}  # {drug_name: [{pmcid, title, pdf_path, indexed}, ...]}
if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = []
if "paper_texts" not in st.session_state:
    st.session_state.paper_texts = {}  # {pmcid: full_text}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_chunks_indexed" not in st.session_state:
    st.session_state.total_chunks_indexed = 0
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False
if "session_namespace" not in st.session_state:
    # Create a unique namespace for this session to isolate vectors
    st.session_state.session_namespace = f"session_{uuid.uuid4().hex[:12]}"

# ----------------------------
# PDF VALIDATION & DOWNLOAD FUNCTIONS
# ----------------------------
def is_valid_pdf(file_path):
    """Return True if file exists, has PDF header, and size > 5KB."""
    try:
        if not os.path.exists(file_path):
            return False

        file_size = os.path.getsize(file_path)
        if file_size < 5000:
            print(f"[WARN] File too small ({file_size} bytes)")
            return False

        with open(file_path, "rb") as f:
            header = f.read(5)
            f.seek(-10, 2)  # Seek to end
            footer = f.read(10)

        has_valid_header = header == b"%PDF-"
        has_valid_footer = b"%%EOF" in footer
        
        if not has_valid_footer:
            print("[WARN] Missing EOF marker")
            return False
            
        return has_valid_header
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        return False


def search_pmc_articles(query, max_results=100):
    """Search PMC for articles matching the query."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        pmc_ids = data.get("esearchresult", {}).get("idlist", [])
        return pmc_ids
    except Exception as e:
        st.error(f"PMC search failed: {e}")
        return []


def get_article_metadata(pmcid):
    """Fetch article title and metadata from PMC."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "json"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        result = data.get("result", {}).get(pmcid, {})
        title = result.get("title", f"PMC{pmcid}")
        return title
    except Exception:
        return f"PMC{pmcid}"


def get_pdf_link_from_pmcid(pmcid):
    """Get PDF link from PMC Open Access API."""
    api_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmcid}"
    try:
        r = requests.get(api_url, timeout=15)
        root = ET.fromstring(r.text)
        for link in root.findall(".//link"):
            if link.attrib.get("format") == "pdf":
                return link.attrib["href"]
    except Exception as e:
        print(f"[ERROR] OA fetch failed for PMC{pmcid}: {e}")
    return None


def download_stream(url, destination, timeout=30):
    """Reliable binary download for HTTP and FTP."""
    if url.startswith("ftp://"):
        with urllib.request.urlopen(url, timeout=timeout) as response, open(destination, "wb") as out:
            shutil.copyfileobj(response, out)
    else:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
            r.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)


def extract_pdf_from_tar_gz(tar_path, output_path):
    """Extract PDF from tar.gz archive."""
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".pdf"):
                    tar.extract(member, path=".")
                    os.rename(member.name, output_path)
                    return True
    except Exception as e:
        print("[ERROR] TAR extraction failed:", e)
    return False


def safe_gunzip(data):
    """Safely decompress gzip data with fallback."""
    try:
        return gzip.decompress(data)
    except Exception as e:
        print(f"[WARN] GZIP decompress failed: {e}")
        return None


def download_pdf(pdf_url, save_path, retries=3):
    """Download PDF with gzip and tar.gz support."""
    for attempt in range(1, retries + 1):
        temp_file = save_path + ".tmp"
        
        try:
            print(f"[INFO] Attempt {attempt}: {pdf_url}")

            download_stream(pdf_url, temp_file)
            time.sleep(0.5)  # Avoid rate limiting

            with open(temp_file, "rb") as f:
                raw = f.read()

            if len(raw) < 100:
                print("[ERROR] Downloaded file too small")
                continue

            # ---- Detect .tar.gz archive ----
            if pdf_url.endswith(".tar.gz") or raw[:2] == b"\x1f\x8b":
                if pdf_url.endswith(".tar.gz"):
                    print("[INFO] Detected TAR.GZ archive. Extracting...")
                    if extract_pdf_from_tar_gz(temp_file, save_path):
                        os.remove(temp_file)
                        if is_valid_pdf(save_path):
                            print("[SUCCESS] Extracted valid PDF")
                            return True
                    print("[SKIP] No valid PDF inside TAR.GZ")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    return False

            # ---- Detect gzip-wrapped PDFs ----
            if raw[:2] == b"\x1f\x8b":
                print("[INFO] Detected GZIPPED content → decompressing")
                decompressed = safe_gunzip(raw)
                if decompressed:
                    raw = decompressed
                else:
                    print("[SKIP] GZIP decompression failed")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    return False

            # ---- Save actual PDF ----
            with open(save_path, "wb") as f:
                f.write(raw)

            if os.path.exists(temp_file):
                os.remove(temp_file)

            if is_valid_pdf(save_path):
                print(f"[SUCCESS] Valid PDF saved: {save_path}")
                return True
            else:
                print("[SKIP] Invalid PDF content")
                if os.path.exists(save_path):
                    os.remove(save_path)
                return False

        except requests.exceptions.Timeout:
            print("[ERROR] Download timeout")
        except requests.exceptions.ConnectionError:
            print("[ERROR] Connection error")
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")

        if os.path.exists(temp_file):
            os.remove(temp_file)

        if attempt < retries:
            print("[INFO] Retrying in 2 seconds...\n")
            time.sleep(2)

    print("[SKIP] Invalid OA PDF link")
    return False


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF with OCR fallback."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)
            page_text = page.get_text()
            
            # If page has very little text, try OCR
            if len(page_text.strip()) < 100:
                try:
                    # Try to extract text from images using PyMuPDF's built-in OCR
                    page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                except Exception:
                    pass
            
            full_text += page_text + "\n"
        
        doc.close()
        
        # Clean up the extracted text
        full_text = clean_extracted_text(full_text)
        
        return full_text
    except Exception as e:
        return f"Error extracting text: {e}"


def clean_extracted_text(text):
    """Clean and filter extracted text to remove noise."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove common PDF artifacts and noise
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            cleaned_lines.append('')
            continue
        
        # Skip lines that are just page numbers
        if re.match(r'^[\d\s\-–—]+$', line):
            continue
        
        # Skip very short lines that are likely headers/footers
        if len(line) < 10 and not any(c.isalpha() for c in line):
            continue
        
        # Skip lines that are just URLs or DOIs
        if line.startswith('http') or line.startswith('doi:') or line.startswith('DOI:'):
            continue
        
        # Skip copyright/license boilerplate
        if any(skip in line.lower() for skip in [
            'creative commons', 'open access', 'terms of the license',
            'licensee', 'copyright ©', 'all rights reserved',
            'downloaded from', 'powered by'
        ]):
            continue
        
        cleaned_lines.append(line)
    
    # Join and clean up
    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()


def is_quality_chunk(chunk):
    """Check if a chunk has meaningful content."""
    if not chunk or len(chunk) < 50:
        return False
    
    # Count actual words (not just tokens)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', chunk)
    if len(words) < 10:
        return False
    
    # Check for too much noise (numbers, special chars)
    alpha_ratio = sum(c.isalpha() for c in chunk) / len(chunk)
    if alpha_ratio < 0.5:
        return False
    
    # Skip chunks that are mostly references/citations
    citation_patterns = [
        r'\[\d+\]',  # [1], [2], etc.
        r'\(\d{4}\)',  # (2024), etc.
        r'et al\.',
        r'doi:',
        r'PMID:',
        r'PMC\d+'
    ]
    citation_count = sum(len(re.findall(p, chunk)) for p in citation_patterns)
    if citation_count > 5 and len(chunk) < 500:
        return False
    
    # Skip chunks that are mostly author names/affiliations
    if any(skip in chunk.lower() for skip in [
        'correspondence:', 'email:', 'affiliation',
        'received:', 'accepted:', 'published:',
        'editor:', 'reviewer'
    ]) and len(chunk) < 300:
        return False
    
    return True


def index_paper_to_pinecone(pmcid, text, drug_name, namespace):
    """Chunk and index paper text to Pinecone using batch embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        return 0
    
    # Filter to quality chunks only
    quality_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if is_quality_chunk(chunk)]
    
    if not quality_chunks:
        print(f"[WARN] No quality chunks found for PMC{pmcid}")
        return 0
    
    print(f"[INFO] PMC{pmcid}: {len(quality_chunks)}/{len(chunks)} quality chunks")
    
    # Batch embed all chunks at once (much faster than one-by-one)
    batch_size = 100
    total_indexed = 0
    
    for batch_start in range(0, len(quality_chunks), batch_size):
        batch = quality_chunks[batch_start:batch_start + batch_size]
        batch_texts = [chunk for _, chunk in batch]
        
        # Batch embedding - single API call for multiple chunks
        vectors = embeddings.embed_documents(batch_texts)
        
        # Prepare batch upsert data
        upsert_data = []
        for (orig_idx, chunk), vector in zip(batch, vectors):
            metadata = {
                "pmcid": str(pmcid),  # Ensure string type
                "drug_name": drug_name,
                "chunk_index": orig_idx,
                "text": chunk
            }
            upsert_data.append((f"{pmcid}_{orig_idx}", vector, metadata))
        
        # Batch upsert to Pinecone with namespace
        index.upsert(vectors=upsert_data, namespace=namespace)
        total_indexed += len(batch)
    
    print(f"[INFO] Indexed {total_indexed} chunks to namespace: {namespace}")
    return total_indexed


def index_paper_parallel(paper_info, drug_name, namespace):
    """Index a single paper - designed to run in parallel."""
    pmcid = paper_info['pmcid']
    text = paper_info.get('text', '')
    
    if not text:
        return pmcid, 0, False
    
    try:
        chunks = index_paper_to_pinecone(pmcid, text, drug_name, namespace)
        return pmcid, chunks, True
    except Exception as e:
        print(f"[ERROR] Failed to index PMC{pmcid}: {e}")
        import traceback
        traceback.print_exc()
        return pmcid, 0, False


def index_papers_parallel(papers_to_index, drug_name, namespace, progress_callback=None):
    """Index multiple papers in parallel using ThreadPoolExecutor."""
    results = {}
    total_chunks = 0
    
    # Use ThreadPoolExecutor for parallel embedding creation
    max_workers = min(4, len(papers_to_index))  # Limit to avoid rate limiting
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all indexing tasks with namespace
        future_to_paper = {
            executor.submit(index_paper_parallel, paper, drug_name, namespace): paper 
            for paper in papers_to_index
        }
        
        completed = 0
        for future in as_completed(future_to_paper):
            paper = future_to_paper[future]
            pmcid, chunks, success = future.result()
            
            results[pmcid] = {'chunks': chunks, 'success': success}
            if success:
                total_chunks += chunks
            
            completed += 1
            if progress_callback:
                progress_callback(completed, len(papers_to_index), pmcid, chunks)
    
    return results, total_chunks


def delete_paper_from_pinecone(pmcid):
    """Delete all chunks for a paper from Pinecone."""
    try:
        namespace = st.session_state.session_namespace
        
        # Delete by prefix (pmcid_0, pmcid_1, etc.)
        deleted_count = 0
        batch_size = 100
        
        for i in range(0, 1000, batch_size):  # Assume max 1000 chunks per paper
            ids_to_delete = [f"{pmcid}_{j}" for j in range(i, i + batch_size)]
            try:
                index.delete(ids=ids_to_delete, namespace=namespace)
                deleted_count += batch_size
            except Exception:
                break
        
        print(f"[INFO] Deleted chunks for PMC{pmcid} from namespace: {namespace}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to delete chunks for PMC{pmcid}: {e}")
        return False


def cleanup_all_indexed_papers():
    """Remove all indexed paper chunks from Pinecone."""
    namespace = st.session_state.session_namespace
    
    try:
        # Delete the entire namespace - most efficient cleanup
        index.delete(delete_all=True, namespace=namespace)
        print(f"[INFO] Deleted all vectors in namespace: {namespace}")
    except Exception as e:
        print(f"[ERROR] Failed to delete namespace: {e}")
    
    # Reset indexed status
    cleaned_pmcids = []
    for drug, papers in st.session_state.retrieved_papers.items():
        for paper in papers:
            if paper.get('indexed', False):
                paper['indexed'] = False
                cleaned_pmcids.append(paper['pmcid'])
    
    return cleaned_pmcids


def get_context_from_selected_papers(query, selected_pmcids):
    """Retrieve relevant context from selected papers only."""
    try:
        # Ensure PMCIDs are strings for consistent filtering
        selected_pmcids_str = [str(pmcid) for pmcid in selected_pmcids]
        namespace = st.session_state.session_namespace
        
        print(f"[INFO] Querying namespace: {namespace} for PMCIDs: {selected_pmcids_str}")
        
        # Query Pinecone directly with namespace and filter
        query_vector = embeddings.embed_query(query)
        
        results = index.query(
            vector=query_vector,
            top_k=50,  # Get more results to filter from
            include_metadata=True,
            namespace=namespace,  # Use session namespace
            filter={"pmcid": {"$in": selected_pmcids_str}}
        )
        
        # Extract text from matching results
        context_chunks = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            chunk_pmcid = str(metadata.get('pmcid', ''))
            chunk_text = metadata.get('text', '')
            
            # Double-check PMCID matches (belt and suspenders)
            if chunk_pmcid in selected_pmcids_str and chunk_text:
                if is_quality_chunk(chunk_text):
                    context_chunks.append({
                        'text': chunk_text,
                        'pmcid': chunk_pmcid,
                        'score': match.get('score', 0)
                    })
        
        print(f"[INFO] Found {len(context_chunks)} matching chunks from namespace {namespace}")
        
        if not context_chunks:
            print(f"[WARN] No matching chunks found for PMCIDs: {selected_pmcids_str} in namespace: {namespace}")
            # Fallback to full text from session state
            context_text = ""
            for pmcid in selected_pmcids:
                paper_text = st.session_state.paper_texts.get(pmcid, "")
                if paper_text:
                    context_text += f"\n\n--- Content from PMC{pmcid} ---\n\n"
                    context_text += paper_text[:10000]
            return context_text if context_text else "No content available for selected papers."
        
        # Sort by relevance score
        context_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        # Ensure each selected paper gets at least some chunks (balanced retrieval)
        num_papers = len(selected_pmcids_str)
        chunks_per_paper = max(5, 15 // num_papers)  # At least 5 chunks per paper
        
        top_chunks = []
        paper_chunk_count = {pmcid: 0 for pmcid in selected_pmcids_str}
        
        # First pass: ensure minimum chunks from each paper
        for chunk in context_chunks:
            pmcid = chunk['pmcid']
            if paper_chunk_count.get(pmcid, 0) < chunks_per_paper:
                top_chunks.append(chunk)
                paper_chunk_count[pmcid] = paper_chunk_count.get(pmcid, 0) + 1
        
        # Second pass: fill with highest scoring remaining chunks
        max_total_chunks = max(20, num_papers * 5)  # Scale with paper count
        remaining = [c for c in context_chunks if c not in top_chunks]
        for chunk in remaining:
            if len(top_chunks) >= max_total_chunks:
                break
            top_chunks.append(chunk)
        
        # Re-sort to ensure top chunks are at the beginning
        top_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        # Debug: show distribution
        paper_counts = {}
        for c in top_chunks:
            paper_counts[c['pmcid']] = paper_counts.get(c['pmcid'], 0) + 1
        print(f"[INFO] Retrieved {len(top_chunks)} chunks - Distribution: {paper_counts}")
        
        # Format context with PMCID attribution
        context_text = "\n\n---\n\n".join([
            f"[From PMC{chunk['pmcid']}]: {chunk['text']}" 
            for chunk in top_chunks
        ])
        
        print(f"[INFO] Retrieved {len(top_chunks)} quality chunks from {len(set(c['pmcid'] for c in top_chunks))} papers")
        return context_text
        
    except Exception as e:
        print(f"[ERROR] Vector retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to full text
        context_text = ""
        for pmcid in selected_pmcids:
            paper_text = st.session_state.paper_texts.get(pmcid, "")
            if paper_text:
                context_text += f"\n\n--- Content from PMC{pmcid} ---\n\n"
                context_text += paper_text[:10000]
        return context_text if context_text else "No content available for selected papers."


def generate_initial_summary(selected_pmcids):
    """Generate an automatic summary of selected papers using the same flow as regular queries."""
    # Precomputed query - treated just like a user question
    precomputed_query = "Give me a comprehensive summary of the research paper(s). Include the title, main objectives, key findings, methodologies used, and conclusions."
    
    # Get context from vector database (same as regular chat)
    context_text = get_context_from_selected_papers(precomputed_query, selected_pmcids)
    
    # Get paper titles for reference
    paper_titles = []
    for drug, papers in st.session_state.retrieved_papers.items():
        for paper in papers:
            if paper['pmcid'] in selected_pmcids:
                paper_titles.append(f"PMC{paper['pmcid']}: {paper['title']}")
    
    # Same prompt structure as regular chat
    prompt = f"""You are a research assistant analyzing scientific research papers related to {', '.join(searched_drugs)}.

You are discussing the following papers (PMCIDs: {', '.join([f'PMC{p}' for p in selected_pmcids])}).
Paper titles:
{chr(10).join(paper_titles)}

Context from selected research papers:
{context_text}

Question: {precomputed_query}

Please provide a detailed, scientific answer based ONLY on the research papers provided. When relevant, highlight findings related to {searched_drugs[0] if searched_drugs else 'the drug'} and any repurposing potential. Cite specific findings and mention which paper (by PMCID) the information comes from."""

    try:
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"


def stream_gemini_response(prompt):
    """Stream response from Gemini model."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error: {e}"


def generate_initial_summary_streaming(selected_pmcids, placeholder):
    """Generate an automatic summary of selected papers with streaming output."""
    # Precomputed query - treated just like a user question
    precomputed_query = "Give me a comprehensive summary of the research paper(s). Include the title, main objectives, key findings, methodologies used, and conclusions."
    
    # Get context from vector database (same as regular chat)
    context_text = get_context_from_selected_papers(precomputed_query, selected_pmcids)
    
    # Get paper titles and drug names for reference
    paper_titles = []
    searched_drugs = list(st.session_state.retrieved_papers.keys())
    for drug, papers in st.session_state.retrieved_papers.items():
        for paper in papers:
            if paper['pmcid'] in selected_pmcids:
                paper_titles.append(f"PMC{paper['pmcid']}: {paper['title']}")
    
    # Same prompt structure as regular chat
    prompt = f"""You are a research assistant analyzing scientific research papers related to {', '.join(searched_drugs)}.

You are discussing the following papers (PMCIDs: {', '.join([f'PMC{p}' for p in selected_pmcids])}).
Paper titles:
{chr(10).join(paper_titles)}

Context from selected research papers:
{context_text}

Question: {precomputed_query}

Please provide a detailed, scientific answer based ONLY on the research papers provided. When relevant, highlight findings related to {searched_drugs[0] if searched_drugs else 'the drug'} and any repurposing potential. Cite specific findings and mention which paper (by PMCID) the information comes from."""

    # Stream the response
    full_response = ""
    for chunk in stream_gemini_response(prompt):
        full_response += chunk
        placeholder.markdown(f"**📋 Summary of Selected Papers**\n\n{full_response}▌")
    
    # Final update without cursor
    placeholder.markdown(f"**📋 Summary of Selected Papers**\n\n{full_response}")
    return full_response


# ----------------------------
# STREAMLIT UI
# ----------------------------

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 20px 0 15px 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #b8daff;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ffeeba;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .step-indicator {
        display: inline-block;
        width: 30px;
        height: 30px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 30px;
        margin-right: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---- App Header ----
st.markdown("# 🔬 Drug Research Paper Chat")
st.markdown("""
<div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
    <h3 style='margin: 0; color: white;'>AI-Powered Drug Repurposing Research Assistant</h3>
    <p style='margin: 10px 0 0 0; opacity: 0.9;'>
        Search PubMed Central for drug repurposing papers, automatically download and index them, 
        then have intelligent conversations with AI about the research findings.
    </p>
</div>
""", unsafe_allow_html=True)

# Pipeline overview
with st.expander("ℹ️ **How This App Works** - Click to expand", expanded=False):
    st.markdown("""
    ### 📋 Pipeline Overview
    
    This application follows a 3-step workflow:
    
    | Step | Action | What Happens |
    |------|--------|--------------|
    | **1️⃣** | **Search & Index** | Search PubMed Central for papers, download PDFs, extract text, and create searchable vector embeddings |
    | **2️⃣** | **Select Papers** | Choose which papers you want to include in your chat context |
    | **3️⃣** | **Chat & Explore** | Ask questions and get AI-powered answers based on the selected papers |
    
    ### 🔧 Technology Stack
    - **PDF Processing**: PyMuPDF for text extraction
    - **Vector Database**: Pinecone for semantic search
    - **Embeddings**: OpenAI text-embedding-ada-002
    - **AI Model**: Google Gemini 2.5 Flash for responses
    - **Data Source**: PubMed Central Open Access
    """)

st.divider()

# ---- Step 1: Drug Search & Auto-Index ----
st.markdown("""
<div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; margin-bottom: 15px;'>
    <h2 style='margin: 0;'><span style='background: #667eea; color: white; padding: 5px 12px; border-radius: 50%; margin-right: 10px;'>1</span> Search & Index Research Papers</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
🔍 **What this step does:**
- Searches PubMed Central for drug repurposing research papers
- Downloads available open-access PDFs
- Extracts and cleans text content from PDFs
- Creates vector embeddings and indexes them for semantic search

> 💡 **Tip**: Start with a specific drug name (e.g., "metformin", "aspirin") for best results.
""")

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    drug_name = st.text_input(
        "🔬 Enter Drug Name:",
        placeholder="e.g., apomorphine, metformin, aspirin",
        help="Enter the name of a drug to search for repurposing research papers"
    )
with col2:
    max_search = st.number_input(
        "📊 Search Pool:",
        min_value=10,
        max_value=200,
        value=100,
        help="Number of articles to search through from PubMed Central. Higher numbers may find more open-access PDFs but take longer."
    )
with col3:
    max_pdfs = st.number_input(
        "📄 Max PDFs:",
        min_value=1,
        max_value=5,
        value=1,
        help="Maximum number of PDFs to download and index. More papers = richer context but longer processing time."
    )

# Search button with better styling
search_col1, search_col2 = st.columns([1, 3])
with search_col1:
    search_button = st.button("🚀 Search & Index", type="primary", use_container_width=True)

if search_button:
    if drug_name:
        # Progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("---")
            st.markdown("### 🔄 Processing Pipeline")
            
            # Phase indicators
            phase_cols = st.columns(3)
            with phase_cols[0]:
                search_status = st.empty()
                search_status.info("🔍 **Phase 1: Searching...**")
            with phase_cols[1]:
                download_status = st.empty()
                download_status.markdown("⏳ Phase 2: Download")
            with phase_cols[2]:
                index_status = st.empty()
                index_status.markdown("⏳ Phase 3: Indexing")
            
            st.markdown("---")
            
            query = f"{drug_name} repurposing"
            pmc_ids = search_pmc_articles(query, max_results=max_search)
            
            search_status.success(f"✅ **Found {len(pmc_ids)} articles**")
            
            if pmc_ids:
                download_status.info("📥 **Phase 2: Downloading...**")
                
                papers = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create temp directory for PDFs
                temp_dir = tempfile.mkdtemp()
                
                download_count = 0
                papers_to_index = []
                
                # Phase 1: Download PDFs and extract text
                for idx, pmcid in enumerate(pmc_ids):
                    if download_count >= max_pdfs:
                        break
                    
                    status_text.text(f"📥 Checking PMC{pmcid}... ({idx+1}/{len(pmc_ids)}) | Downloaded: {download_count}/{max_pdfs}")
                    progress_bar.progress((idx + 1) / len(pmc_ids) * 0.5)
                    
                    pdf_url = get_pdf_link_from_pmcid(pmcid)
                    if not pdf_url:
                        continue
                    
                    title = get_article_metadata(pmcid)
                    
                    paper_info = {
                        "pmcid": pmcid,
                        "title": title,
                        "pdf_url": pdf_url,
                        "pdf_path": None,
                        "indexed": False,
                        "has_pdf": False,
                        "chunks": 0
                    }
                    
                    save_path = os.path.join(temp_dir, f"PMC{pmcid}.pdf")
                    if download_pdf(pdf_url, save_path):
                        paper_info["pdf_path"] = save_path
                        paper_info["has_pdf"] = True
                        download_count += 1
                        
                        # Extract text
                        text = extract_text_from_pdf(save_path)
                        st.session_state.paper_texts[pmcid] = text
                        paper_info["text"] = text
                        
                        papers.append(paper_info)
                        papers_to_index.append(paper_info)
                    
                    time.sleep(0.3)
                
                download_status.success(f"✅ **Downloaded {download_count} PDFs**")
                
                # Phase 2: Parallel indexing
                total_chunks = 0
                if papers_to_index:
                    index_status.info("🔢 **Phase 3: Indexing...**")
                    status_text.text(f"🔢 Creating vector embeddings for {len(papers_to_index)} papers...")
                    
                    def update_progress(completed, total, pmcid, chunks):
                        progress = 0.5 + (completed / total * 0.5)
                        progress_bar.progress(progress)
                        status_text.text(f"🔢 Indexed PMC{pmcid} ({chunks} chunks) - {completed}/{total} papers")
                    
                    results, total_chunks = index_papers_parallel(
                        papers_to_index, drug_name, st.session_state.session_namespace, update_progress
                    )
                    
                    for paper in papers:
                        if paper['pmcid'] in results:
                            paper['indexed'] = results[paper['pmcid']]['success']
                            paper['chunks'] = results[paper['pmcid']]['chunks']
                    
                    st.session_state.total_chunks_indexed += total_chunks
                    index_status.success(f"✅ **Indexed {total_chunks} chunks**")
                
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.retrieved_papers[drug_name] = papers
                
                # Results summary
                st.markdown("---")
                if download_count == 0:
                    st.warning(f"⚠️ No valid open-access PDFs found in {len(pmc_ids)} articles. Try a different drug name or increase the search pool.")
                else:
                    st.success(f"""
                    ### ✅ Processing Complete!
                    
                    | Metric | Value |
                    |--------|-------|
                    | 📄 PDFs Downloaded | {download_count} |
                    | 🔢 Text Chunks Created | {total_chunks} |
                    | ⚡ Processing Mode | Parallel |
                    
                    **Next Step:** Select papers below and start chatting!
                    """)
            else:
                st.warning("❌ No papers found. Try a different drug name.")
    else:
        st.warning("⚠️ Please enter a drug name to search.")

# ---- Step 2: Select Papers for Chat ----
if st.session_state.retrieved_papers:
    st.divider()
    
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin-bottom: 15px;'>
        <h2 style='margin: 0;'><span style='background: #28a745; color: white; padding: 5px 12px; border-radius: 50%; margin-right: 10px;'>2</span> Select Papers for Chat</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    📋 **What this step does:**
    - Review the downloaded and indexed papers
    - Select which papers to include in your chat context
    - More papers = broader knowledge base, but may dilute focus
    
    > 💡 **Tip**: For focused answers, select 1-2 highly relevant papers. For comprehensive analysis, select all.
    """)
    
    for drug, papers in st.session_state.retrieved_papers.items():
        st.markdown(f"### 📚 Papers for: **{drug}**")
        
        # Summary metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("📄 Total Papers", len(papers))
        with metric_cols[1]:
            indexed_count = sum(1 for p in papers if p.get('indexed', False))
            st.metric("✅ Indexed", indexed_count)
        with metric_cols[2]:
            total_paper_chunks = sum(p.get('chunks', 0) for p in papers)
            st.metric("🔢 Total Chunks", total_paper_chunks)
        with metric_cols[3]:
            selected_count = sum(1 for p in papers if p['pmcid'] in st.session_state.selected_papers)
            st.metric("☑️ Selected", selected_count)
        
        # Papers table
        df_data = []
        for paper in papers:
            status_icon = "✅" if paper.get('indexed', False) else "❌"
            selected_icon = "☑️" if paper['pmcid'] in st.session_state.selected_papers else "⬜"
            df_data.append({
                "Select": selected_icon,
                "Status": status_icon,
                "PMCID": f"PMC{paper['pmcid']}",
                "Title": paper["title"][:70] + "..." if len(paper["title"]) > 70 else paper["title"],
                "Chunks": paper.get("chunks", 0)
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Selection controls
        st.markdown("**📌 Quick Selection:**")
        col_sel1, col_sel2, col_sel3 = st.columns([1, 1, 2])
        with col_sel1:
            if st.button(f"✅ Select All", key=f"select_all_{drug}", use_container_width=True):
                for paper in papers:
                    if paper['pmcid'] not in st.session_state.selected_papers:
                        st.session_state.selected_papers.append(paper['pmcid'])
                st.rerun()
        with col_sel2:
            if st.button(f"❌ Clear All", key=f"deselect_all_{drug}", use_container_width=True):
                for paper in papers:
                    if paper['pmcid'] in st.session_state.selected_papers:
                        st.session_state.selected_papers.remove(paper['pmcid'])
                st.rerun()
        
        # Individual paper selection
        st.markdown("**📝 Individual Selection:**")
        cols = st.columns(2)
        for idx, paper in enumerate(papers):
            col = cols[idx % 2]
            with col:
                chunks_badge = f"({paper.get('chunks', 0)} chunks)" if paper.get('chunks', 0) > 0 else "(not indexed)"
                key = f"select_{drug}_{paper['pmcid']}"
                is_selected = st.checkbox(
                    f"**PMC{paper['pmcid']}** {chunks_badge}\n{paper['title'][:45]}...",
                    key=key,
                    value=paper['pmcid'] in st.session_state.selected_papers
                )
                
                if is_selected and paper['pmcid'] not in st.session_state.selected_papers:
                    st.session_state.selected_papers.append(paper['pmcid'])
                elif not is_selected and paper['pmcid'] in st.session_state.selected_papers:
                    st.session_state.selected_papers.remove(paper['pmcid'])

# ---- Step 3: View Papers & Chat ----
if st.session_state.selected_papers:
    st.divider()
    
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; margin-bottom: 15px;'>
        <h2 style='margin: 0;'><span style='background: #dc3545; color: white; padding: 5px 12px; border-radius: 50%; margin-right: 10px;'>3</span> Chat with Your Papers</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    💬 **What this step does:**
    - Uses RAG (Retrieval-Augmented Generation) to find relevant passages
    - Sends context + your question to Google Gemini AI
    - Provides cited, paper-specific answers
    
    > 💡 **Tip**: Ask specific questions like "What are the mechanisms of action?" or "What diseases was [drug] tested against?"
    """)
    
    # Selected papers summary
    st.markdown(f"""
    <div style='background-color: #d4edda; padding: 15px; border-radius: 8px; border: 1px solid #c3e6cb; margin: 10px 0;'>
        <strong>📚 Chat Context:</strong> {len(st.session_state.selected_papers)} paper(s) selected | 
        <strong>🔢 Searchable Chunks:</strong> ~{st.session_state.total_chunks_indexed} | 
        <strong>🤖 AI Model:</strong> Gemini 2.5 Flash
    </div>
    """, unsafe_allow_html=True)
    
    # Collapsible paper viewer with PDF display
    with st.expander("📄 **View Selected Papers** (Click to expand)", expanded=False):
        for pmcid in st.session_state.selected_papers:
            # Find the paper info
            pdf_path = None
            paper_title = f"PMC{pmcid}"
            for drug, papers in st.session_state.retrieved_papers.items():
                for paper in papers:
                    if paper['pmcid'] == pmcid:
                        pdf_path = paper.get('pdf_path')
                        paper_title = paper.get('title', f"PMC{pmcid}")
                        break
            
            st.markdown(f"#### 📖 PMC{pmcid}")
            st.markdown(f"**Title:** {paper_title}")
            
            tab1, tab2 = st.tabs(["📄 PDF View", "📝 Text View"])
            
            with tab1:
                if pdf_path and os.path.exists(pdf_path):
                    try:
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="⬇️ Download PDF",
                                data=pdf_file,
                                file_name=f"PMC{pmcid}.pdf",
                                mime="application/pdf",
                                key=f"download_{pmcid}"
                            )
                    except Exception as e:
                        st.error(f"Could not read PDF: {e}")
                    
                    try:
                        with open(pdf_path, "rb") as f:
                            sanitized_key = f"pdf_view_{pmcid}".replace("_", "-").replace("/", "-").replace("\\", "-")
                            st.pdf(f, height=500, key=sanitized_key)
                    except Exception as e:
                        st.warning(f"PDF preview unavailable: {e}")
                else:
                    st.info("PDF file not available for preview.")
            
            with tab2:
                if pmcid in st.session_state.paper_texts:
                    text = st.session_state.paper_texts[pmcid]
                    st.text_area(
                        "Extracted Text:",
                        value=text[:8000] + "\n\n... [truncated for display]" if len(text) > 8000 else text,
                        height=300,
                        key=f"content_{pmcid}"
                    )
                else:
                    st.warning("Text not available for this paper.")
            
            st.divider()
    
    # Chat interface
    st.markdown("### 💬 AI Research Assistant")
    
    # Start Chat button
    if not st.session_state.chat_initialized:
        st.markdown("""
        <div style='background-color: #fff3cd; padding: 15px; border-radius: 8px; border: 1px solid #ffeeba; margin: 10px 0;'>
            <strong>🚀 Ready to start!</strong> Click the button below to generate an initial summary of your selected papers 
            and begin the conversation.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Chat & Generate Summary", type="primary", use_container_width=True):
            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                placeholder.markdown("**📋 Analyzing selected papers...**\n\n_Extracting key information..._")
                summary = generate_initial_summary_streaming(st.session_state.selected_papers, placeholder)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"**📋 Summary of Selected Papers**\n\n{summary}"
            })
            st.session_state.chat_initialized = True
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(msg['content'])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg['content'])
        st.markdown("---")
    
    # Chat input and controls
    if st.session_state.chat_initialized:
        # Example questions
        with st.expander("💡 **Example Questions** (Click for suggestions)", expanded=False):
            st.markdown("""
            Try asking:
            - "What are the main findings about [drug name]'s repurposing potential?"
            - "What mechanisms of action are discussed?"
            - "What diseases or conditions were studied?"
            - "What were the experimental methods used?"
            - "Are there any clinical trials mentioned?"
            - "What are the limitations of this research?"
            - "Compare the findings across the selected papers"
            """)
        
        user_question = st.chat_input("Ask a question about the selected papers...")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            clear_btn = st.button("🗑️ Clear Chat", use_container_width=True)
        with col2:
            new_summary_btn = st.button("📋 New Summary", use_container_width=True)
        with col3:
            end_session_btn = st.button("🔚 End Session", type="secondary", use_container_width=True)
        
        if clear_btn:
            st.session_state.chat_history = []
            st.session_state.chat_initialized = False
            st.rerun()
        
        if new_summary_btn:
            st.session_state.chat_history = []
            st.session_state.chat_initialized = False
            st.rerun()
        
        if end_session_btn:
            with st.spinner("🧹 Cleaning up vector database..."):
                cleaned = cleanup_all_indexed_papers()
                if cleaned:
                    st.success(f"✅ Removed {len(cleaned)} papers from vector database")
                
                st.session_state.chat_history = []
                st.session_state.selected_papers = []
                st.session_state.total_chunks_indexed = 0
                st.session_state.chat_initialized = False
                time.sleep(1)
                st.rerun()
        
        if user_question:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_question)
            
            selected_pmcids = st.session_state.selected_papers
            context_text = get_context_from_selected_papers(user_question, selected_pmcids)
            searched_drugs = list(st.session_state.retrieved_papers.keys())
            
            prompt = f"""You are a research assistant analyzing scientific research papers related to {', '.join(searched_drugs)}.

You are discussing the following papers (PMCIDs: {', '.join([f'PMC{p}' for p in selected_pmcids])}).

Context from selected research papers:
{context_text}

Question: {user_question}

Please provide a detailed, scientific answer based ONLY on the research papers provided. When relevant, focus on findings related to {searched_drugs[0] if searched_drugs else 'the drug'} and any repurposing potential. Cite specific findings and mention which paper (by PMCID) the information comes from."""

            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                full_response = ""
                for chunk in stream_gemini_response(prompt):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                answer = full_response
            
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.info("👆 Click **'Start Chat & Generate Summary'** above to begin your research conversation.")

# ---- Sidebar: Status & Controls ----
with st.sidebar:
    st.markdown("## 📊 Session Dashboard")
    
    st.markdown("---")
    
    # Status metrics with better visualization
    st.markdown("### 📈 Current Status")
    
    total_papers = sum(len(p) for p in st.session_state.retrieved_papers.values())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Papers", total_papers, help="Total papers downloaded and indexed")
    with col2:
        st.metric("🔢 Chunks", st.session_state.total_chunks_indexed, help="Total text chunks in vector database")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("☑️ Selected", len(st.session_state.selected_papers), help="Papers selected for chat context")
    with col4:
        st.metric("💬 Messages", len(st.session_state.chat_history), help="Messages in current chat")
    
    # Chat status indicator
    if st.session_state.chat_initialized:
        st.success("🟢 Chat Active")
    else:
        st.info("🔵 Chat Ready")
    
    st.markdown("---")
    
    # Session info
    st.markdown("### 🔐 Session Info")
    st.caption(f"**Session ID:** `{st.session_state.session_namespace[:12]}...`")
    st.caption("Each session uses isolated vector storage for privacy.")
    
    st.markdown("---")
    
    # Control buttons
    st.markdown("### ⚙️ Controls")
    
    if st.button("🔄 Reset Everything", use_container_width=True, help="Clear all data and start fresh"):
        cleanup_all_indexed_papers()
        st.session_state.session_namespace = f"session_{uuid.uuid4().hex[:12]}"
        st.session_state.retrieved_papers = {}
        st.session_state.selected_papers = []
        st.session_state.paper_texts = {}
        st.session_state.chat_history = []
        st.session_state.total_chunks_indexed = 0
        st.session_state.chat_initialized = False
        st.rerun()
    
    if total_papers > 0:
        if st.button("🧹 Clear Vector DB", use_container_width=True, help="Remove indexed papers from database"):
            with st.spinner("Cleaning..."):
                cleaned = cleanup_all_indexed_papers()
                st.session_state.total_chunks_indexed = 0
                st.session_state.chat_initialized = False
                st.success(f"Cleaned {len(cleaned)} papers")
                st.rerun()
    
    st.markdown("---")
    
    # Help section
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        **Quick Tips:**
        - 🔍 Use specific drug names
        - 📄 Start with 1-2 papers for focused answers
        - 💬 Ask specific questions
        - 🧹 Clean up when done
        
        **Troubleshooting:**
        - No PDFs? Try increasing search pool
        - Slow indexing? Reduce max PDFs
        - Poor answers? Check if papers are relevant
        """)