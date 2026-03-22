import os
import time
from pathlib import Path

import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx
import pandas as pd
import streamlit as st

# -------------------------------
# PIPELINE IMPORTS
# -------------------------------
from pipeline.parser import extract_text_from_pdf, extract_chapter_4
from pipeline.chunker import chunk_text
from pipeline.llm_handler import extract_from_chunk
from pipeline.extractor import aggregate_results
from pipeline.formatter import to_dataframe
from pipeline.validator import validate_df

# NEW: Import your generic CSV handler
from chatbot.main import get_intent_and_filters, retrieve_data, generate_final_response, handle_generic_csv

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Document Processor", layout="wide", initial_sidebar_state="collapsed")

EOC_FOLDER = "./EOC"
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "output.csv"

# -------------------------------
# UI STYLES
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #f4f4f4; }
#MainMenu, footer, header { visibility: hidden; }

.top-bar {
    background: #111111; padding: 16px 32px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex; align-items: center; gap: 14px;
}
.top-bar .logo-mark {
    width: 26px; height: 26px; border: 2px solid #ffffff;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; color: #fff; letter-spacing: 0.5px; flex-shrink: 0;
}
.top-bar h1 {
    font-size: 14px !important; font-weight: 500 !important;
    color: #ffffff !important; margin: 0 !important; letter-spacing: 0.3px;
}
.top-bar .tag {
    margin-left: auto; font-size: 11px; font-family: 'IBM Plex Mono', monospace;
    color: #666; letter-spacing: 0.5px; text-transform: uppercase;
}

.section-label {
    font-size: 10px; font-weight: 600; letter-spacing: 1.4px;
    text-transform: uppercase; color: #999999; margin-bottom: 12px;
    font-family: 'IBM Plex Mono', monospace;
}

.card {
    background: #ffffff; border: 1px solid #e2e2e2;
    border-radius: 4px; padding: 24px 28px; margin-bottom: 20px;
}
.card-header {
    font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
    color: #999; font-family: 'IBM Plex Mono', monospace;
    border-bottom: 1px solid #eeeeee; padding-bottom: 12px; margin-bottom: 20px;
}

.stButton > button {
    background-color: #111111 !important; color: #ffffff !important;
    border: none !important; border-radius: 3px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    letter-spacing: 0.4px !important; height: 42px !important;
    padding: 0 28px !important; cursor: pointer;
    transition: background-color 0.15s ease !important;
}
.stButton > button:hover { background-color: #333333 !important; }
.stButton > button:active { background-color: #000000 !important; }

[data-testid="stDownloadButton"] > button {
    background-color: #ffffff !important; color: #111111 !important;
    border: 1.5px solid #333333 !important; border-radius: 3px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    height: 42px !important; padding: 0 24px !important;
    letter-spacing: 0.3px !important; transition: background-color 0.15s ease !important;
}
[data-testid="stDownloadButton"] > button:hover { background-color: #f4f4f4 !important; }

.status-banner {
    display: flex; align-items: center; gap: 10px; padding: 11px 18px;
    border-radius: 3px; font-size: 13px; font-weight: 500; margin-bottom: 20px;
}
.status-banner.success { background: #f0f7f0; border: 1px solid #b8d8b8; color: #2d5c2d; }
.status-banner.running { background: #f5f5f5; border: 1px solid #d8d8d8; color: #555; }
.status-banner.error   { background: #fdf2f2; border: 1px solid #e8c4c4; color: #7a2d2d; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.status-dot.green { background: #3a8c3a; }
.status-dot.red   { background: #c0392b; }
.status-dot.pulse { background: #111111; animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

.timer-wrap { display: flex; align-items: baseline; gap: 8px; margin: 10px 0 16px 0; }
.timer-box { font-family: 'IBM Plex Mono', monospace; font-size: 32px; font-weight: 500; color: #111111; letter-spacing: 1px; }
.timer-unit { font-size: 13px; color: #aaa; font-family: 'IBM Plex Mono', monospace; }

[data-testid="stProgressBar"] > div > div { background-color: #111111 !important; border-radius: 2px !important; }
[data-testid="stProgressBar"] > div { background-color: #e0e0e0 !important; border-radius: 2px !important; height: 4px !important; }

.metrics-row { display: flex; gap: 14px; margin-bottom: 24px; }
.metric-card { flex: 1; background: #ffffff; border: 1px solid #e4e4e4; border-radius: 4px; padding: 16px 20px; }
.metric-card .m-label { font-size: 10px; font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.8px; text-transform: uppercase; color: #999; margin-bottom: 6px; }
.metric-card .m-value { font-size: 22px; font-weight: 600; color: #111111; line-height: 1; }
.metric-card .m-sub   { font-size: 11px; color: #bbb; margin-top: 4px; }

.file-list { display: flex; flex-direction: column; gap: 0; }
.file-row { display: flex; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid #f2f2f2; font-size: 13px; }
.file-row:last-child { border-bottom: none; }
.file-row .fname { font-family: 'IBM Plex Mono', monospace; font-size: 12px; flex: 1; color: #333; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.file-badge { font-size: 10px; font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.5px; padding: 3px 9px; border-radius: 3px; font-weight: 500; flex-shrink: 0; }
.badge-done    { background: #eaf4ea; color: #2d6e2d; }
.badge-running { background: #f0f0f0; color: #777; }
.badge-pending { background: #f7f7f7; color: #bbb; }
.badge-error   { background: #fdf2f2; color: #c0392b; }

[data-testid="stDataFrame"] { border: 1px solid #e0e0e0 !important; border-radius: 4px !important; }
[data-testid="stDataFrame"] table { font-size: 13px !important; font-family: 'IBM Plex Sans', sans-serif !important; }
[data-testid="stDataFrame"] th { background-color: #f4f4f4 !important; color: #555555 !important; font-weight: 600 !important; font-size: 11px !important; letter-spacing: 0.5px !important; text-transform: uppercase !important; border-bottom: 1px solid #ddd !important; }

.chat-wrap { max-height: 400px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; margin-bottom: 16px; padding-right: 6px; }
.chat-wrap::-webkit-scrollbar { width: 3px; }
.chat-wrap::-webkit-scrollbar-thumb { background: #ddd; border-radius: 3px; }

.msg { display: flex; gap: 10px; align-items: flex-start; }
.msg.user { flex-direction: row-reverse; }
.msg-avatar { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 600; flex-shrink: 0; font-family: 'IBM Plex Mono', monospace; }
.msg.user .msg-avatar { background: #111111; color: #fff; }
.msg.bot  .msg-avatar { background: #eeeeee; color: #666; }
.msg-body { display: flex; flex-direction: column; max-width: 74%; }
.msg.user .msg-body { align-items: flex-end; }
.msg-bubble { padding: 10px 14px; border-radius: 4px; font-size: 13px; line-height: 1.65; }
.msg.user .msg-bubble { background: #111111; color: #ffffff; border-radius: 4px 4px 0 4px; }
.msg.bot  .msg-bubble { background: #f0f0f0; color: #222222; border-radius: 4px 4px 4px 0; }
.msg-time { font-size: 10px; color: #ccc; margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }
.chat-empty { text-align: center; padding: 36px 0; color: #ccc; font-size: 13px; }

.rule { border: none; border-top: 1px solid #e4e4e4; margin: 28px 0; }

/* --- ADDED: Context Selector Styling --- */
div[role="radiogroup"] label span:first-child { display: none !important; }
div[role="radiogroup"] label p { color: #888888 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; margin-right: 25px !important; transition: none !important;}
div[role="radiogroup"] [data-checked="true"] p { color: #000000 !important; text-decoration: underline !important; text-underline-offset: 6px; font-weight: 600 !important; }
div[role="radiogroup"] label:hover p { color: #000000 !important; }
</style>

<div class="top-bar">
    <div class="logo-mark">DP</div>
    <h1>Document Processor</h1>
    <span class="tag">Batch Pipeline v1.0</span>
</div>
""", unsafe_allow_html=True)


# -------------------------------
# SESSION STATE (Added CSV Tracker)
# -------------------------------
for key, default in {
    "run_pipeline": False,
    "pipeline_done": False,
    "final_df": None,
    "elapsed": None,
    "file_count": 0,
    "chat_history": [],
    "uploaded_df": None,        # NEW
    "uploaded_filename": None   # NEW
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -------------------------------
# HELPERS
# -------------------------------
def get_pdf_files(folder_path):
    if not os.path.exists(folder_path):
        return []
    return sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ])

def process_single_file(file_path):
    try:
        print("inside very first !!")
        chapter4_text = extract_chapter_4(file_path)
        chunks = chunk_text(chapter4_text)
        results = []
        for chunk in chunks:
            output = extract_from_chunk(chunk)
            results.append(output)
        structured_data = aggregate_results(results)
        filename = Path(file_path).stem
        df = to_dataframe(structured_data, filename)
        return df
    except Exception as e:
        st.error(f"Error in process_single_file for {file_path}: {e}")
        return None

def run_batch_pipeline(folder_path):
    file_paths = get_pdf_files(folder_path)
    if not file_paths:
        return None, 0

    total = len(file_paths)
    all_dfs = []
    
    timer_slot = st.empty()
    overall_bar = st.progress(0)
    filelist_slot = st.empty()

    stop_timer = False
    global_start_time = time.time()

    def update_timer():
        while not stop_timer:
            elapsed = time.time() - global_start_time
            try:
                timer_slot.markdown(
                    f'<div class="timer-wrap">'
                    f'<div class="timer-box">{elapsed:0.1f}</div>'
                    f'<div class="timer-unit">seconds elapsed</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            except:
                break
            time.sleep(0.1) 

    def render_file_list(current_idx):
        rows = ""
        for j, fp in enumerate(file_paths):
            name = Path(fp).name
            status = 'done' if j < current_idx else ('running' if j == current_idx else 'pending')
            badge_class = f"badge-{status}"
            rows += f'<div class="file-row"><span class="fname">{name}</span><span class="file-badge {badge_class}">{status}</span></div>'
        
        filelist_slot.markdown(
            f'<div class="card" style="padding:14px 20px;margin-bottom:0;"><div class="file-list">{rows}</div></div>',
            unsafe_allow_html=True
        )

    timer_thread = threading.Thread(target=update_timer, daemon=True)
    add_script_run_ctx(timer_thread) 
    timer_thread.start()

    try:
        for i, file_path in enumerate(file_paths):
            render_file_list(i)
            df = process_single_file(file_path)
            if df is not None:
                all_dfs.append(df)
            overall_bar.progress((i + 1) / total)
    finally:
        stop_timer = True
        timer_thread.join(timeout=1.0)

    total_elapsed = round(time.time() - global_start_time, 2)
    overall_bar.empty()
    timer_slot.empty()
    filelist_slot.empty()

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True), total_elapsed
    return None, total_elapsed

# ================================
# SECTION 1 — PIPELINE
# ================================
st.markdown('<div class="section-label">Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-header">Batch Extraction — EOC Folder</div>', unsafe_allow_html=True)

col_btn, col_del, col_desc = st.columns([1, 1, 3])
with col_btn:
    run_clicked = st.button("▶  Run Pipeline", use_container_width=True)
with col_del:
    del_clicked = st.button("🗑 Delete Output", use_container_width=True)
with col_desc:
    pdf_count = len(get_pdf_files(EOC_FOLDER))
    st.markdown(
        f'<p style="font-size:13px;color:#999;margin:10px 0 0 0;">'
        f'Processes all PDFs in '
        f'<code style="background:#f0f0f0;padding:2px 6px;border-radius:3px;font-size:12px;">{EOC_FOLDER}</code>'
        f' &nbsp;·&nbsp; <strong style="color:#555;">{pdf_count} file{"s" if pdf_count != 1 else ""} found</strong>.'
        f'</p>',
        unsafe_allow_html=True
    )

if del_clicked:
    if CSV_PATH.exists():
        os.remove(CSV_PATH)
    st.session_state.final_df = None
    st.session_state.pipeline_done = False
    st.rerun()

if run_clicked:
    st.session_state.run_pipeline = True
    st.session_state.pipeline_done = False
    st.session_state.final_df = None

st.markdown('</div>', unsafe_allow_html=True)

# ---- Running ----
if st.session_state.run_pipeline and not st.session_state.pipeline_done:
    st.markdown(
        '<div class="status-banner running">'
        '<div class="status-dot pulse"></div>'
        '<span>Batch pipeline running — do not close this window</span>'
        '</div>',
        unsafe_allow_html=True
    )

    final_df, elapsed_time = run_batch_pipeline(EOC_FOLDER)

    if final_df is not None:
        final_df.to_csv(CSV_PATH, index=False)
        st.session_state.final_df = final_df
        st.session_state.elapsed = elapsed_time
        st.session_state.file_count = len(get_pdf_files(EOC_FOLDER))
        st.session_state.pipeline_done = True
        st.session_state.run_pipeline = False
        st.rerun()
    else:
        st.markdown(
            '<div class="status-banner error">'
            '<div class="status-dot red"></div>'
            '<span>No data extracted. Check that the EOC folder contains valid PDFs.</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.session_state.run_pipeline = False

# ---- Results (Metrics Only) ----
if st.session_state.pipeline_done and st.session_state.final_df is not None:
    df          = st.session_state.final_df
    elapsed     = st.session_state.elapsed
    file_count  = st.session_state.file_count

    st.markdown(
        f'<div class="status-banner success">'
        f'<div class="status-dot green"></div>'
        f'<span>Batch complete — {file_count} file{"s" if file_count != 1 else ""} processed in <strong>{elapsed}s</strong></span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="metrics-row">'
        f'  <div class="metric-card"><div class="m-label">Files processed</div>'
        f'    <div class="m-value">{file_count}</div><div class="m-sub">PDF documents</div></div>'
        f'  <div class="metric-card"><div class="m-label">Total rows</div>'
        f'    <div class="m-value">{len(df)}</div><div class="m-sub">extracted records</div></div>'
        f'  <div class="metric-card"><div class="m-label">Columns</div>'
        f'    <div class="m-value">{len(df.columns)}</div><div class="m-sub">fields</div></div>'
        f'  <div class="metric-card"><div class="m-label">Elapsed</div>'
        f'    <div class="m-value">{elapsed}s</div><div class="m-sub">wall clock</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="rule">', unsafe_allow_html=True)


# ================================
# SECTION 1.5 — DATA CONTEXT & UPLOAD
# ================================
st.markdown('<div class="section-label">Data Context</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col_ctx, col_up = st.columns([1, 1])

with col_up:
    st.markdown('<div class="card-header" style="border:none; padding:0; margin-bottom:10px;">Upload External CSV</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.session_state.uploaded_filename != uploaded_file.name:
            st.session_state.uploaded_df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_filename = uploaded_file.name
            st.rerun()
    else:
        if st.session_state.uploaded_filename is not None:
            st.session_state.uploaded_df = None
            st.session_state.uploaded_filename = None
            st.rerun()

with col_ctx:
    st.markdown('<div class="card-header" style="border:none; padding:0; margin-bottom:10px;">Select Target for Chatbot</div>', unsafe_allow_html=True)
    
    options = ["None"]
    if os.path.exists(CSV_PATH) or st.session_state.final_df is not None:
        options.append("Pipeline output")
    if st.session_state.uploaded_df is not None:
        options.append("csv upload")
    
    selected_context = st.radio("Context", options, label_visibility="collapsed", horizontal=True)

# ---- DYNAMIC PREVIEW RENDERING ----
if selected_context == "csv upload" and st.session_state.uploaded_df is not None:
    st.markdown('<div style="margin-top:20px; font-size:12px; font-weight:600; color:#666; font-family:\'IBM Plex Mono\';">PREVIEW: External CSV</div>', unsafe_allow_html=True)
    st.dataframe(st.session_state.uploaded_df, use_container_width=True, height=400)

elif selected_context == "Pipeline output":
    # Always load df if context is selected, regardless of run state
    pipe_df = st.session_state.final_df
    if pipe_df is None and CSV_PATH.exists():
        pipe_df = pd.read_csv(CSV_PATH)
        
    if pipe_df is not None:
        st.markdown('<div style="margin-top:20px; font-size:12px; font-weight:600; color:#666; font-family:\'IBM Plex Mono\';">PREVIEW: Pipeline Output</div>', unsafe_allow_html=True)
        st.dataframe(pipe_df, use_container_width=True, height=400, column_config={col: st.column_config.Column(width="large") for col in pipe_df.columns})
        
        col_dl, _ = st.columns([1, 5])
        with col_dl:
            with open(CSV_PATH, "rb") as f:
                st.download_button("↓  Download CSV", f, file_name="output.csv", mime="text/csv", key="dl_pipe_csv")

st.markdown('</div>', unsafe_allow_html=True)


# ================================
# SECTION 2 — CHATBOT
# ================================
# def process_chat_query(user_input, df=None):
#     if df is None:
#         if os.path.exists(CSV_PATH):
#             df = pd.read_csv(CSV_PATH)
#         else:
#             return "Error: No data available."
#     print("inside here")
#     intent = get_intent_and_filters(user_input, df.columns.tolist())
#     relevant_data = retrieve_data(df, intent)
#     return generate_final_response(user_input, relevant_data)
def process_chat_query(user_input, df=None):
    if df is None:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
        else:
            return "Error: No data file found."

    intent = get_intent_and_filters(user_input, df.columns.tolist())
    relevant_data, category = retrieve_data(df, intent)
    return generate_final_response(user_input, relevant_data, category)


st.markdown('<div class="section-label">Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-header">Chat with your document</div>', unsafe_allow_html=True)

if st.session_state.chat_history:
    html = '<div class="chat-wrap">'
    for msg in st.session_state.chat_history:
        role, content, ts = msg["role"], msg["content"], msg.get("time", "")
        if role == "user":
            html += (
                f'<div class="msg user"><div class="msg-avatar">You</div>'
                f'<div class="msg-body"><div class="msg-bubble">{content}</div>'
                f'<div class="msg-time">{ts}</div></div></div>'
            )
        elif role == "bot":
            html += (
                f'<div class="msg bot"><div class="msg-avatar">AI</div>'
                f'<div class="msg-body"><div class="msg-bubble">{content}</div>'
                f'<div class="msg-time">{ts}</div></div></div>'
            )
        elif role == "result":
            html += (
                f'<div class="msg bot"><div class="msg-avatar">AI</div>'
                f'<div class="msg-body"><div class="msg-bubble" style="padding:6px 8px;">'
                f'<span style="font-size:11px;font-family:\'IBM Plex Mono\',monospace;color:#999;">'
                f'Result — {ts}</span></div></div></div>'
            )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "result" and msg.get("dataframe") is not None:
            st.dataframe(
                msg["dataframe"],
                use_container_width=True,
                column_config={col: st.column_config.Column(width="large") for col in msg["dataframe"].columns}
            )

else:
    st.markdown(
        '<div class="chat-empty">No messages yet. Ask a question about the extracted data.</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<style>
    .thinking span { animation: blink 1.4s infinite both; font-size: 24px; font-weight: bold; }
    .thinking span:nth-child(2) { animation-delay: .2s; }
    .thinking span:nth-child(3) { animation-delay: .4s; }
    @keyframes blink { 0% { opacity: .2; } 20% { opacity: 1; } 100% { opacity: .2; } }
</style>
""", unsafe_allow_html=True)

if "processing" not in st.session_state:
    st.session_state.processing = False

if st.session_state.processing:
    st.markdown(
        f'<div class="msg bot"><div class="msg-avatar">AI</div>'
        f'<div class="msg-body"><div class="msg-bubble thinking">'
        f'<span>.</span><span>.</span><span>.</span></div></div></div>',
        unsafe_allow_html=True
    )

# The 'disabled' parameter prevents double-sending AND locks it if "None" is selected
input_disabled = st.session_state.processing or (selected_context == "None")
user_input = st.chat_input("Ask something about the data…", disabled=input_disabled)

if user_input:
    st.session_state.processing = True
    ts = time.strftime("%H:%M")
    st.session_state.chat_history.append({"role": "user", "content": user_input, "time": ts})
    st.rerun()

if st.session_state.processing:
    last_query = st.session_state.chat_history[-1]["content"]
    
    if selected_context == "Pipeline output":
        df_context = st.session_state.final_df if st.session_state.get('pipeline_done') else (pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else None)
        if df_context is not None:
            result = process_chat_query(last_query, df=df_context)
        else:
            result = "Error: Pipeline output data not found."
            
    elif selected_context == "csv upload":
        df_context = st.session_state.uploaded_df
        if df_context is not None:
            result = handle_generic_csv(last_query, df_context)
            print("Result is: ", result)
        else:
            result = "Error: Uploaded CSV data is missing."
    else:
        result = "Please select a data context."

    curr_ts = time.strftime("%H:%M")
    if isinstance(result, str):
        st.session_state.chat_history.append({"role": "bot", "content": result, "time": curr_ts})
    
    elif isinstance(result, pd.DataFrame):
        st.session_state.chat_history.append({"role": "bot", "content": "Here is the data for your query:", "time": curr_ts})
        st.session_state.chat_history.append({"role": "result", "dataframe": result, "time": curr_ts})
    
    elif isinstance(result, dict):
        if result.get("message"):
            st.session_state.chat_history.append({"role": "bot", "content": result["message"], "time": curr_ts})
        if result.get("dataframe") is not None:
            st.session_state.chat_history.append({"role": "result", "dataframe": result["dataframe"], "time": curr_ts})

    st.session_state.processing = False
    st.rerun()