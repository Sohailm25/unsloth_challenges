import modal

image = modal.Image.debian_slim().pip_install("torch")
app = modal.App(image=image)

@app.function(gpu="T4")
def run():
    import torch

    assert torch.cuda.is_available()
