import json
from pathlib import Path
import time
from config import PARSED_DIR
from pipeline_01_ingest import chunk_markdown

def rechunk_all():
    jsonl_files = sorted(PARSED_DIR.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} parsed files to re-chunk semantically.\n")

    for file_path in jsonl_files:
        print(f"> Processing {file_path.name}...")
        t0 = time.time()

        # Read existing chunks
        existing_chunks = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_chunks.append(json.loads(line))
        
        if not existing_chunks:
            print("  Empty file, skipping.")
            continue

        existing_chunks.sort(key=lambda x: x["chunk_index"])
        
        # Reconstruct markdown
        md_text = "\n\n".join(c["text"] for c in existing_chunks)
        
        doc_id = existing_chunks[0]["doc_id"]
        source_file = existing_chunks[0]["source_file"]

        try:
            new_chunks = chunk_markdown(md_text, doc_id, source_file)
            
            # Write back
            with open(file_path, "w", encoding="utf-8") as f:
                for c in new_chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"  [OK] {len(new_chunks)} semantic chunks generated | {time.time()-t0:.1f}s")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [ERROR] FAILED: {e}")

if __name__ == "__main__":
    rechunk_all()
    print("Semantic chunking complete!")
