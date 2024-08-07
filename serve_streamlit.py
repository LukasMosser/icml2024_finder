# Based on Modal Labs example script for serving streamlit apps
import shlex
import subprocess
from pathlib import Path
import modal

# ## Define container dependencies
#
# The `app.py` script imports three third-party packages, so we include these in the example's
# image definition.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit~=1.35.0",
        "numpy~=1.26.4",
        "pandas~=2.2.2",
        "huggingface_hub",
        "lancedb",
        "openai",
        "tantivy",
        "jsonlines",
        "cohere",
        "streamlit_js_eval",
    )
    .copy_local_dir("icml_finder", "/root/icml_finder")
)

app = modal.App(name="icml-finder", image=image)

# ## Mounting the `app.py` script
#
# We can just mount the `app.py` script inside the container at a pre-defined path using a Modal
# [`Mount`](https://modal.com/docs/guide/local-data#mounting-directories).

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = Path("/root/app.py")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)

# ## Spawning the Streamlit server
#
# Inside the container, we will run the Streamlit server in a background subprocess using
# `subprocess.Popen`. We also expose port 8000 using the `@web_server` decorator.
# Here we also include a volume that has the vectordb and the two secrets for cohere and openai


@app.function(
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount],
    secrets=[
        modal.Secret.from_name("icml-finder-openai"),
        modal.Secret.from_name("cohere-api-key"),
    ],
    volumes={"/icml_data": modal.Volume.from_name("icml_data")},
)
@modal.web_server(8000)
def run():
    target = shlex.quote(str(streamlit_script_remote_path))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)


# ## Iterate and Deploy
#
# While you're iterating on your screamlit app, you can run it "ephemerally" with `modal serve`. This will
# run a local process that watches your files and updates the app if anything changes.
#
# ```shell
# modal serve serve_streamlit.py
# ```
#
# Once you're happy with your changes, you can deploy your application with
#
# ```shell
# modal deploy serve_streamlit.py
# ```
#
# If successful, this will print a URL for your app, that you can navigate to from
# your browser 🎉 .
