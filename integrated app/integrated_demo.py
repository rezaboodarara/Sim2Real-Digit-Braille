import multiprocessing as mp
import os
import collections
import collections.abc
from pathlib import Path
import numpy as np
import threading
import time

# Patch collections for older modules
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.Iterable = collections.abc.Iterable
collections.Iterator = collections.abc.Iterator


# ── 1. ISOLATED TENSORFLOW BACKGROUND WORKER ──────────────────────────────────
def tf_worker(conn, model_path):
    """
    This function runs in a completely separate Linux process.
    It traps TensorFlow so it cannot collide with PyTorch/Open3D.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    import tensorflow as tf
    
    # Keras Monkeypatches
    original_bn_init = tf.keras.layers.BatchNormalization.__init__
    original_dense_init = tf.keras.layers.Dense.__init__

    def patched_bn_init(self, *args, **kwargs):
        kwargs.pop('renorm', None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        original_bn_init(self, *args, **kwargs)

    def patched_dense_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None) 
        original_dense_init(self, *args, **kwargs)

    tf.keras.layers.BatchNormalization.__init__ = patched_bn_init
    tf.keras.layers.Dense.__init__ = patched_dense_init

    # Load Model
    model = tf.keras.models.load_model(model_path)
    print("✓ Background Worker: TensorFlow Keras model loaded securely.")

    # Infinite loop to listen for images sent from the Gradio UI
    while True:
        try:
            tensor = conn.recv()
            if tensor is None:
                break
            probs = model.predict(tensor, verbose=0)[0]
            conn.send(probs) # Send results back to the UI
        except EOFError:
            break


# ── 2. MAIN APP LAUNCHER (PYTORCH + GRADIO) ───────────────────────────────────
if __name__ == "__main__":
    # MUST be the absolute first thing called to isolate the C++ memory
    mp.set_start_method('spawn', force=True)
    
    base_path = Path(__file__).parent.parent.resolve()
    asl_model_path = str(base_path / "best_model_limited.keras")
    
    # Create the invisible memory pipe and boot up the isolated TF process
    parent_conn, child_conn = mp.Pipe()
    tf_proc = mp.Process(target=tf_worker, args=(child_conn, asl_model_path), daemon=True)
    tf_proc.start()

    # NOW it is safe to load the PyTorch and Open3D stack
    import cv2
    import torch
    import hydra
    import gradio as gr
    from digit_depth.third_party import geom_utils
    from digit_depth.digit import DigitSensor
    from digit_depth.train.prepost_mlp import preproc_mlp, post_proc_mlp
    from digit_depth.handlers import find_recent_model

    IMAGE_SIZE  = (64, 64)
    CLASSES     = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    TOP_K       = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @hydra.main(config_path=f"{base_path}/config", config_name="digit.yaml", version_base=None)
    def launch_app(cfg):
        print("\n--- Loading PyTorch Depth Model ---")
        model_path = find_recent_model(f"{base_path}/models")
        depth_model = torch.load(model_path, weights_only=False).to(device)
        depth_model.eval()
        print("✓ PyTorch Model loaded.")

        print("\n--- Initializing DIGIT Sensor ---")
        digit = DigitSensor(cfg.sensor.fps, cfg.sensor.resolution, cfg.sensor.serial_num)
        digit_call = digit()
        
        print("\nCalibrating zero depth... (Do NOT touch the sensor for a few seconds)")
        dm_zero = 0
        for _ in range(50):
            frame = digit_call.get_frame()
            img_np = preproc_mlp(frame)
            img_np = depth_model(img_np).detach().cpu().numpy()
            img_np, _ = post_proc_mlp(img_np)
            
            gradx_img, grady_img = geom_utils._normal_to_grad_depth(
                img_normal=img_np, gel_width=cfg.sensor.gel_width,
                gel_height=cfg.sensor.gel_height, bg_mask=None)
            img_depth = geom_utils._integrate_grad_depth(
                gradx_img, grady_img, boundary=None, bg_mask=None, max_depth=cfg.max_depth)
            
            dm_zero += img_depth.detach().cpu().numpy()
            
        dm_zero = dm_zero / 50.0
        print("Calibration complete! Starting web interface...\n")


        # --- CONTINUOUS CAMERA THREAD ---
        latest_frame = np.zeros((64, 64, 3), dtype=np.uint8) # Safe default

        def process_depth(frame):
            """Helper function to calculate depth natively."""
            img_np = preproc_mlp(frame)
            img_np = depth_model(img_np).detach().cpu().numpy()
            img_np, _ = post_proc_mlp(img_np)
            
            gradx_img, grady_img = geom_utils._normal_to_grad_depth(
                img_normal=img_np, gel_width=cfg.sensor.gel_width, gel_height=cfg.sensor.gel_height, bg_mask=None)
            img_depth = geom_utils._integrate_grad_depth(
                gradx_img, grady_img, boundary=None, bg_mask=None, max_depth=cfg.max_depth)
            img_depth = img_depth.detach().cpu().numpy() 
            
            diff = np.abs(img_depth - dm_zero)
            AMPLIFY = 150000 
            diff_scaled = np.clip(diff * AMPLIFY, 0, 255).astype(np.uint8)
            _, sharp_img = cv2.threshold(diff_scaled, 30, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(sharp_img, cv2.COLOR_GRAY2RGB)

        def camera_worker():
            """Runs forever in the background, keeping the USB buffer empty and fresh."""
            nonlocal latest_frame
            while True:
                try:
                    frame = digit_call.get_frame()
                    latest_frame = process_depth(frame)
                except Exception as e:
                    print(f"Camera thread error: {e}")
                    time.sleep(1)

        # Start the background thread
        threading.Thread(target=camera_worker, daemon=True).start()

        # --- UI FUNCTIONS ---
        def stream_to_ui():
            """Continuously yields the latest frame to the Gradio web interface."""
            while True:
                yield latest_frame
                time.sleep(0.05) # Caps stream at 20 FPS to save bandwidth

        def capture_image():
            """Grabs the current frame from the live stream."""
            return latest_frame
            
        def clear_capture():
            """Clears the captured image and resets the prediction boxes."""
            return None, "", {}

        def predict_via_worker(image: np.ndarray):
            """Sends the captured image to the TensorFlow background process."""
            if image is None: 
                return "Please capture an image first.", {}
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            tensor = (resized.astype("float32") / 255.0).reshape(1, 64, 64, 1)
            
            parent_conn.send(tensor)
            probs = parent_conn.recv()

            top_indices  = np.argsort(probs)[::-1][:TOP_K]
            top_classes  = [CLASSES[i] for i in top_indices]
            top_probs    = [float(probs[i]) for i in top_indices]

            top_label    = f"{top_classes[0]}  —  {top_probs[0]*100:.1f}% confidence"
            bar_data     = {cls: prob for cls, prob in zip(top_classes, top_probs)}
            return top_label, bar_data

        # --- GRADIO LAYOUT ---
        with gr.Blocks(title="Unified Tactile Sensor App") as demo:
            gr.Markdown("# ✋ Unified DIGIT Sensor & Braille Classifier")
            
            with gr.Row():
                # LEFT COLUMN: Live Feed
                with gr.Column(scale=1):
                    live_feed = gr.Image(label="Live DIGIT Feed", interactive=False)
                    capture_btn = gr.Button("📸 Capture Frame", variant="primary")
                    
                # RIGHT COLUMN: Captured Image & Prediction
                with gr.Column(scale=1):
                    captured_img = gr.Image(label="Captured Image", type="numpy", interactive=False)
                    
                    with gr.Row():
                        predict_btn = gr.Button("🔍 Predict Letter", variant="secondary")
                        clear_btn = gr.Button("❌ Cancel / Clear", variant="stop")
                        
                    label_out = gr.Textbox(label="Top Prediction", interactive=False, lines=1)
                    bar_out = gr.Label(label=f"Top-{TOP_K} probabilities", num_top_classes=TOP_K)

            # --- UI EVENT WIRING ---
            # Triggers the live video stream the moment the web page opens
            demo.load(fn=stream_to_ui, inputs=[], outputs=[live_feed])
            
            capture_btn.click(fn=capture_image, inputs=[], outputs=[captured_img])
            predict_btn.click(fn=predict_via_worker, inputs=[captured_img], outputs=[label_out, bar_out])
            clear_btn.click(fn=clear_capture, inputs=[], outputs=[captured_img, label_out, bar_out])

        # Launch Server
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)

    launch_app()