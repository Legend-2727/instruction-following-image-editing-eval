import streamlit as st
import sqlite3
import json
import os
import datetime
from pathlib import Path
import pandas as pd

# Define default taxonomy structure (you can update this or load from schema)
TAXONOMY_LABELS = [
    "Wrong Object",
    "Missing Object",
    "Extra Object",
    "Wrong Attribute",
    "Spatial Error",
    "Style Mismatch",
    "Over-editing",
    "Under-editing",
    "Artifact / Quality Issue",
    "Ambiguous Prompt",
    "Failed Removal"
]

DB_PATH = "data/splits/annotations.db"
POOL_PATH = "data/splits/human_taxonomy_pool.jsonl"
IMAGE_ROOT = "data/hf_snapshots/xlingual_picobanana_full"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            sample_id TEXT,
            annotator_id TEXT,
            labels TEXT,
            comments TEXT,
            unclear BOOLEAN,
            timestamp TEXT,
            UNIQUE(sample_id, annotator_id)
        )
    ''')
    conn.commit()
    return conn

@st.cache_data
def load_pool(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_annotations(conn):
    return pd.read_sql_query("SELECT * FROM annotations", conn)

def save_annotation(conn, sample_id, annotator_id, labels, comments, unclear):
    c = conn.cursor()
    # Upsert logic
    c.execute('''
        INSERT INTO annotations (sample_id, annotator_id, labels, comments, unclear, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(sample_id, annotator_id) 
        DO UPDATE SET 
            labels=excluded.labels, 
            comments=excluded.comments, 
            unclear=excluded.unclear, 
            timestamp=excluded.timestamp
    ''', (sample_id, annotator_id, json.dumps(labels), comments, unclear, datetime.datetime.utcnow().isoformat()))
    conn.commit()

def export_data(conn):
    df = get_annotations(conn)
    export_dir = Path("data/splits")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Export CSV
    csv_path = export_dir / "annotations_export.csv"
    df.to_csv(csv_path, index=False)
    
    # Export JSONL
    jsonl_path = export_dir / "annotations_export.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.to_dict()
            record["labels"] = json.loads(record["labels"])
            f.write(json.dumps(record) + "\n")
            
    return csv_path, jsonl_path

def main():
    st.set_page_config(layout="wide", page_title="Taxonomy Annotator")
    st.title("Concurrent Taxonomy Annotator 🏷️")
    
    # Init DB
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    conn = init_db()
    
    pool = load_pool(POOL_PATH)
    if not pool:
        st.error(f"Pool file not found at {POOL_PATH}. Please run the split preparation script first.")
        return

    # Sidebar: Login & Filter
    st.sidebar.header("Session")
    annotator_id = st.sidebar.text_input("Annotator ID (Your Name/Initial):", value="").strip()
    
    if not annotator_id:
        st.warning("Please enter your Annotator ID in the sidebar to start labeling.")
        return
        
    st.sidebar.markdown("---")
    st.sidebar.header("Filters")
    df_annotations = get_annotations(conn)
    
    filter_type = st.sidebar.radio("Show samples:", [
        "Unlabeled", 
        "Labeled by Me", 
        "Labeled by Anyone",
        "All Samples"
    ])
    
    # Apply filters
    labeled_by_anyone = set(df_annotations["sample_id"].unique())
    labeled_by_me = set(df_annotations[df_annotations["annotator_id"] == annotator_id]["sample_id"].unique())
    
    if filter_type == "Unlabeled":
        filtered_pool = [s for s in pool if s["id"] not in labeled_by_anyone]
    elif filter_type == "Labeled by Me":
        filtered_pool = [s for s in pool if s["id"] in labeled_by_me]
    elif filter_type == "Labeled by Anyone":
        filtered_pool = [s for s in pool if s["id"] in labeled_by_anyone]
    else:
        filtered_pool = pool
        
    st.sidebar.write(f"Showing **{len(filtered_pool)}** of {len(pool)} total samples.")
    
    # Export UI
    st.sidebar.markdown("---")
    if st.sidebar.button("Export Data"):
        csv_path, jsonl_path = export_data(conn)
        st.sidebar.success(f"Exported to:\n{csv_path}\n{jsonl_path}")

    # Navigation
    if not filtered_pool:
        st.info("No samples match the selected filter.")
        return
        
    idx = st.sidebar.number_input("Sample Index", min_value=0, max_value=len(filtered_pool)-1, value=0)
    st.sidebar.progress((idx + 1) / len(filtered_pool))
    
    sample = filtered_pool[idx]
    s_id = sample["id"]
    
    # Load existing user annotation if any
    existing_row = df_annotations[(df_annotations["sample_id"] == s_id) & (df_annotations["annotator_id"] == annotator_id)]
    
    default_labels = []
    default_comments = ""
    default_unclear = False
    
    if not existing_row.empty:
        rec = existing_row.iloc[0]
        try:
            default_labels = json.loads(rec["labels"])
        except:
            pass
        default_comments = rec["comments"]
        default_unclear = bool(rec["unclear"])
    
    # Check if others annotated
    others = df_annotations[(df_annotations["sample_id"] == s_id) & (df_annotations["annotator_id"] != annotator_id)]
    if not others.empty:
        st.info(f"💡 This sample has been labeled by {len(others)} other annotator(s).")
    
    # Main UI
    st.markdown(f"### Sample: `{s_id}` | Edit Type: `{sample.get('edit_type', 'N/A')}`")
    
    col1, col2 = st.columns(2)
    with col1:
        src_path = os.path.join(IMAGE_ROOT, sample.get('source_path', ''))
        st.image(src_path, caption="Source Image", use_container_width=True)
    with col2:
        tgt_path = os.path.join(IMAGE_ROOT, sample.get('target_path', ''))
        st.image(tgt_path, caption="Edited Image", use_container_width=True)
        
    # Prompts
    st.markdown("#### Instructions")
    st.info(f"**🇬🇧 EN:** {sample.get('instruction_en', 'N/A')}")
    
    with st.expander("Show Hindi & Bangla instructions"):
        st.markdown(f"**🇮🇳 HI:** {sample.get('instruction_hi', 'N/A')}")
        st.markdown(f"**🇧🇩 BN:** {sample.get('instruction_bn', 'N/A')}")
        
    # Annotation Form
    st.markdown("---")
    st.markdown("### Annotation")
    
    selected_labels = st.multiselect(
        "Select Taxonomy Error Labels (Multi-select):", 
        options=TAXONOMY_LABELS,
        default=[l for l in default_labels if l in TAXONOMY_LABELS]
    )
    
    comments = st.text_area("Comments / Notes:", value=default_comments)
    unclear = st.checkbox("Mark as Unclear / Skip for Discussion", value=default_unclear)
    
    col_save, col_next = st.columns([1, 4])
    with col_save:
        if st.button("Save Annotation", type="primary"):
            save_annotation(conn, s_id, annotator_id, selected_labels, comments, unclear)
            st.success("Saved!")
            # Trigger experimental rerun to load the next state logically, or user advances index
            st.rerun()

if __name__ == "__main__":
    main()
