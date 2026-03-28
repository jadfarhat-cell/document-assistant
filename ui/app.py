import gradio as gr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import settings

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0a0a0f",
    body_background_fill_dark="#0a0a0f",
    background_fill_primary="#101014",
    background_fill_primary_dark="#101014",
    background_fill_secondary="#18181c",
    background_fill_secondary_dark="#18181c",
    border_color_primary="#27272a",
    border_color_primary_dark="#27272a",
    body_text_color="#fafafa",
    body_text_color_dark="#fafafa",
    body_text_color_subdued="#a1a1aa",
    body_text_color_subdued_dark="#a1a1aa",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_dark="#2563eb",
    button_primary_background_fill_hover="#3b82f6",
    button_primary_background_fill_hover_dark="#3b82f6",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#18181c",
    button_secondary_background_fill_dark="#18181c",
    button_secondary_background_fill_hover="#27272a",
    button_secondary_background_fill_hover_dark="#27272a",
    button_secondary_text_color="#fafafa",
    button_secondary_text_color_dark="#fafafa",
    input_background_fill="#18181c",
    input_background_fill_dark="#18181c",
    input_border_color="#27272a",
    input_border_color_dark="#27272a",
    input_border_color_focus="#2563eb",
    input_border_color_focus_dark="#2563eb",
    block_background_fill="#101014",
    block_background_fill_dark="#101014",
    block_border_color="#27272a",
    block_border_color_dark="#27272a",
    block_label_background_fill="#18181c",
    block_label_background_fill_dark="#18181c",
    block_label_text_color="#a1a1aa",
    block_label_text_color_dark="#a1a1aa",
    block_title_text_color="#fafafa",
    block_title_text_color_dark="#fafafa",
    shadow_drop="0 4px 6px -1px rgba(0, 0, 0, 0.4)",
    shadow_drop_lg="0 10px 15px -3px rgba(0, 0, 0, 0.5)",
    block_shadow="none",
    block_shadow_dark="none",
)

custom_css = ""

pipeline = RAGPipeline(
    collection_name=settings.chroma_collection,
    persist_directory=settings.chroma_persist_dir,
    model=settings.ollama_model,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    top_k=settings.top_k,
)


def check_status():
    stats = pipeline.get_stats()
    return str(stats)


def upload_file(file):
    if file is None:
        return "No file selected"
    try:
        result = pipeline.ingest_file(file.name)
        return f"Ingested: {result['filename']}, chunks: {result['chunks']}"
    except Exception as e:
        return f"Error: {str(e)}"


def query_documents(question, top_k):
    if not question:
        return "Please enter a question.", ""
    try:
        result = pipeline.query(question, top_k=int(top_k), stream=False)
        sources_text = "**Sources:**\n"
        for source in result["sources"]:
            sources_text += f"- {source}\n"
        return result["answer"], sources_text
    except Exception as e:
        return f"Error: {str(e)}", ""


def clear_index():
    pipeline.clear()
    return "Cleared all indexed documents."


with gr.Blocks(title="RAG Document Assistant") as demo:
    with gr.Tabs():
        with gr.Tab("Chat"):
            question_input = gr.Textbox(label="Your Question", lines=2)
            top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top K")
            query_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Textbox(label="Answer", lines=10, interactive=False)
            sources_output = gr.Markdown(label="Sources")
            query_btn.click(query_documents, inputs=[question_input, top_k_slider], outputs=[answer_output, sources_output])

        with gr.Tab("Upload"):
            file_input = gr.File(label="Select a document")
            upload_btn = gr.Button("Upload File", variant="primary")
            upload_output = gr.Textbox(label="Status", interactive=False)
            upload_btn.click(upload_file, inputs=file_input, outputs=upload_output)

        with gr.Tab("Settings"):
            clear_btn = gr.Button("Clear All Documents", variant="stop")
            clear_output = gr.Textbox(label="Status", interactive=False)
            clear_btn.click(clear_index, outputs=clear_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, theme=theme, css=custom_css)
