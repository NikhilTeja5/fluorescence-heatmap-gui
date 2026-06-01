import os
import cv2
import zipfile
import shutil
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


# Page setup

st.set_page_config(
    page_title="Fluorescence Heatmap GUI",
    layout="wide"
)

st.title("Fluorescence Microscopy Particle Heatmap GUI")

st.write(
    "Upload fluorescence microscopy images or a ZIP file. "
    "Generate enhanced particle/molecule heatmaps with selectable colors."
)


# Folders

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_results"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Sidebar controls

st.sidebar.header("Heatmap Settings")

strip_width = st.sidebar.slider(
    "Strip width",
    200,
    1000,
    420
)

strip_height = st.sidebar.slider(
    "Strip height",
    60,
    400,
    120
)

heatmap_color = st.sidebar.selectbox(
    "Select heatmap color",
    [
        "Blue",
        "Cyan Blue",
        "Yellow",
        "Green",
        "Red",
        "Magenta",
        "Fire",
        "Inferno",
        "Jet"
    ]
)

background_kernel = st.sidebar.slider(
    "Background removal kernel",
    21,
    151,
    61,
    step=2
)

denoise_kernel = st.sidebar.selectbox(
    "Denoise kernel",
    [3, 5, 7],
    index=0
)

particle_kernel = st.sidebar.slider(
    "Particle enhancement kernel",
    5,
    61,
    17,
    step=2
)

low_cut_percentile = st.sidebar.slider(
    "Background suppression percentile",
    40,
    95,
    70
)

high_percentile = st.sidebar.slider(
    "Brightness clipping percentile",
    95.0,
    99.9,
    99.8
)

gamma = st.sidebar.slider(
    "Particle brightness gamma",
    0.2,
    1.5,
    0.45
)


# Upload

uploaded_files = st.file_uploader(
    "Upload microscopy images or ZIP file",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "zip"],
    accept_multiple_files=True
)


# Helper functions

def clear_folder(folder):
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)


def collect_images(input_dir):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths = []

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))

    return sorted(paths)


def apply_heatmap_color(heat_uint8, color_name):
    """
    Converts grayscale heat intensity into selectable heatmap color.
    """

    if color_name == "Blue":
        heatmap_rgb = np.zeros(
            (heat_uint8.shape[0], heat_uint8.shape[1], 3),
            dtype=np.uint8
        )
        heatmap_rgb[:, :, 2] = heat_uint8
        return heatmap_rgb

    elif color_name == "Cyan Blue":
        heatmap_rgb = np.zeros(
            (heat_uint8.shape[0], heat_uint8.shape[1], 3),
            dtype=np.uint8
        )
        heatmap_rgb[:, :, 1] = (heat_uint8 * 0.6).astype(np.uint8)
        heatmap_rgb[:, :, 2] = heat_uint8
        return heatmap_rgb

    elif color_name == "Yellow":
        heatmap_rgb = np.zeros(
            (heat_uint8.shape[0], heat_uint8.shape[1], 3),
            dtype=np.uint8
        )
        heatmap_rgb[:, :, 0] = heat_uint8
        heatmap_rgb[:, :, 1] = heat_uint8
        return heatmap_rgb

    elif color_name == "Green":
        heatmap_rgb = np.zeros(
            (heat_uint8.shape[0], heat_uint8.shape[1], 3),
            dtype=np.uint8
        )
        heatmap_rgb[:, :, 1] = heat_uint8
        return heatmap_rgb

    elif color_name == "Red":
        heatmap_rgb = np.zeros(
            (heat_uint8.shape[0], heat_uint8.shape[1], 3),
            dtype=np.uint8
        )
        heatmap_rgb[:, :, 0] = heat_uint8
        return heatmap_rgb

    elif color_name == "Magenta":
        heatmap_rgb = np.zeros(
            (heat_uint8.shape[0], heat_uint8.shape[1], 3),
            dtype=np.uint8
        )
        heatmap_rgb[:, :, 0] = heat_uint8
        heatmap_rgb[:, :, 2] = heat_uint8
        return heatmap_rgb

    elif color_name == "Fire":
        heatmap_bgr = cv2.applyColorMap(
            heat_uint8,
            cv2.COLORMAP_HOT
        )
        return cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB
        )

    elif color_name == "Inferno":
        heatmap_bgr = cv2.applyColorMap(
            heat_uint8,
            cv2.COLORMAP_INFERNO
        )
        return cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB
        )

    elif color_name == "Jet":
        heatmap_bgr = cv2.applyColorMap(
            heat_uint8,
            cv2.COLORMAP_JET
        )
        return cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB
        )

    else:
        return cv2.cvtColor(
            cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB
        )


def make_particle_heatmap(
    image_path,
    strip_size=(420, 120),
    heatmap_color="Cyan Blue",
    background_kernel=61,
    denoise_kernel=3,
    particle_kernel=17,
    low_cut_percentile=70,
    high_percentile=99.8,
    gamma=0.45
):
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    image_rgb = cv2.resize(
        image_rgb,
        strip_size
    )

    green = image_rgb[:, :, 1]

    # Mild denoising
    green_denoised = cv2.medianBlur(
        green,
        denoise_kernel
    )

    # Background removal
    if background_kernel % 2 == 0:
        background_kernel += 1

    background = cv2.GaussianBlur(
        green_denoised,
        (background_kernel, background_kernel),
        0
    )

    bg_removed = cv2.subtract(
        green_denoised,
        background
    )

    # Top-hat particle enhancement
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (particle_kernel, particle_kernel)
    )

    particles = cv2.morphologyEx(
        bg_removed,
        cv2.MORPH_TOPHAT,
        kernel
    ).astype(np.float32)

    # Suppress weak background texture
    low_cut = np.percentile(
        particles,
        low_cut_percentile
    )

    particles[particles < low_cut] = 0

    # Normalize strong particles
    high = np.percentile(
        particles,
        high_percentile
    )

    heat = np.clip(
        particles / (high + 1e-8),
        0,
        1
    )

    # Gamma correction
    heat = np.power(
        heat,
        gamma
    )

    heat_uint8 = (
        heat * 255
    ).astype(np.uint8)

    # Smooth for fluorescence-like appearance
    heat_uint8 = cv2.GaussianBlur(
        heat_uint8,
        (0, 0),
        sigmaX=0.7
    )

    heatmap_rgb = apply_heatmap_color(
        heat_uint8,
        heatmap_color
    )

    return image_rgb, heatmap_rgb


def create_strip_panel(
    image_paths,
    save_path,
    strip_size=(420, 120),
    heatmap_color="Cyan Blue"
):
    n = len(image_paths)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(10, 1.45 * n)
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, img_path in enumerate(image_paths):

        original, heatmap = make_particle_heatmap(
            img_path,
            strip_size=strip_size,
            heatmap_color=heatmap_color,
            background_kernel=background_kernel,
            denoise_kernel=denoise_kernel,
            particle_kernel=particle_kernel,
            low_cut_percentile=low_cut_percentile,
            high_percentile=high_percentile,
            gamma=gamma
        )

        axes[i, 0].imshow(original)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap)
        axes[i, 1].axis("off")

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=0.03,
        hspace=0.08
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()



# Main app


if uploaded_files:

    clear_folder(INPUT_DIR)

    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            INPUT_DIR,
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.name.lower().endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(INPUT_DIR)

    image_paths = collect_images(INPUT_DIR)

    st.success(f"Images found: {len(image_paths)}")

    if len(image_paths) > 0:

        st.subheader("Single Image Preview")

        sample_original, sample_heatmap = make_particle_heatmap(
            image_paths[0],
            strip_size=(strip_width, strip_height),
            heatmap_color=heatmap_color,
            background_kernel=background_kernel,
            denoise_kernel=denoise_kernel,
            particle_kernel=particle_kernel,
            low_cut_percentile=low_cut_percentile,
            high_percentile=high_percentile,
            gamma=gamma
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                sample_original,
                caption="Original Green Fluorescence",
                use_container_width=True
            )

        with col2:
            st.image(
                sample_heatmap,
                caption=f"{heatmap_color} Particle Heatmap",
                use_container_width=True
            )

        if st.button("Generate Full Strip Panel"):

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{heatmap_color.lower().replace(' ', '_')}_particle_heatmap_panel.png"
            )

            create_strip_panel(
                image_paths=image_paths,
                save_path=output_path,
                strip_size=(strip_width, strip_height),
                heatmap_color=heatmap_color
            )

            st.success("Panel generated successfully!")

            st.image(
                output_path,
                use_container_width=True
            )

            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Final Panel",
                    data=f,
                    file_name=os.path.basename(output_path),
                    mime="image/png"
                )

else:
    st.info("Upload images or a ZIP file to begin.")