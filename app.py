import torch
import gradio as gr

from nst.transform_net import TransformNet
from nst.inference import stylize
from nst.config import get_style_configs

def stylize_pil_image(content_image, style_name, models):
    return stylize(models[style_name], content_image, scaling_width=1024)

def select_style(selection: gr.SelectData):
    """
    Style Selector Callback
    """
    return selection.value['caption']  # returns the style name (e.g., "Starry Night")

def create_app():
    with gr.Blocks() as app:
        gr.Markdown("## 🎨 Neural Style Transfer")

        with gr.Row():
            with gr.Column():
                content_input = gr.Image(label="Upload your content image", type="pil", height=500)

                style_gallery = gr.Gallery(
                    label="Choose a style by clicking the image",
                    value=thumbnails,
                    show_label=True,
                    interactive=False,
                    columns=len(thumbnails),
                    # rows=2,
                    height=250
                )

                selected_style = gr.State(value='Starry Night')  # default
                style_gallery.select(fn=select_style, outputs=selected_style)

                submit_btn = gr.Button("Stylize")

            with gr.Column():
                result_output = gr.Image(label="Stylized Result", type="pil", height=500)

        submit_btn.click(
            fn=stylize_pil_image,
            inputs=[content_input, selected_style, models],
            outputs=result_output
        )

    return app


style_configs = get_style_configs()
thumbnails = [(cfg.style_image_path, cfg.style_name) for cfg in style_configs]  # (img_path, caption)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}
for cfg in style_configs:
    model = TransformNet().to(device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.eval()
    models[cfg.style_name] = model

app = create_app()
app.launch(debug=True)