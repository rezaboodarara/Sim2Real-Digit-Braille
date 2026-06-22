""" Visualizes the calculated depth map using OpenCV with a Heatmap. """
import cv2
import hydra
import torch
import numpy as np
from pathlib import Path
from digit_depth.third_party import geom_utils
from digit_depth.digit import DigitSensor
from digit_depth.train.prepost_mlp import *
from digit_depth.handlers import find_recent_model

seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = Path(__file__).parent.parent.resolve()

@hydra.main(config_path=f"{base_path}/config", config_name="digit.yaml", version_base=None)
def show_depth(cfg):
    model_path = find_recent_model(f"{base_path}/models")
    model = torch.load(model_path).to(device)
    model.eval()

    digit = DigitSensor(cfg.sensor.fps, cfg.sensor.resolution, cfg.sensor.serial_num)
    digit_call = digit()
    dm_zero_counter = 0
    dm_zero = 0
    
    print("Starting depth map pipeline. Press 'q' in the image window to quit.")
    
    while True:
        frame = digit_call.get_frame()
        img_np = preproc_mlp(frame)
        img_np = model(img_np).detach().cpu().numpy()
        img_np, _ = post_proc_mlp(img_np)
        
        gradx_img, grady_img = geom_utils._normal_to_grad_depth(img_normal=img_np, gel_width=cfg.sensor.gel_width,
                                                                gel_height=cfg.sensor.gel_height,bg_mask=None)
        img_depth = geom_utils._integrate_grad_depth(gradx_img, grady_img, boundary=None, bg_mask=None,max_depth=cfg.max_depth)
        img_depth = img_depth.detach().cpu().numpy() 
        
        if dm_zero_counter < 50:
            dm_zero += img_depth
            dm_zero_counter += 1
            if dm_zero_counter == 1:
                print("Calibrating zero depth... (Do not touch the sensor for a few seconds)")
            continue
        elif dm_zero_counter == 50:
            dm_zero = dm_zero/50
            dm_zero_counter += 1
            print("Calibration complete! Showing depth map.")
        
        # --- NEW VISUALIZATION LOGIC ---
        
        # 1. Take absolute difference so both positive/negative deformations show up
        diff = np.abs(img_depth - dm_zero)
        
        # 2. Amplify the signal. 
        # TUNE THIS: If it's still too dark, increase to 5000. If it's all white noise, lower to 500.
        AMPLIFY = 3000 
        diff_scaled = np.clip(diff * AMPLIFY, 0, 255).astype(np.uint8)

        # 3. Apply a heatmap color filter (Blue = flat gel, Red/Yellow = deep indentation)
        color_depth = cv2.applyColorMap(diff_scaled, cv2.COLORMAP_JET)

        cv2.imshow("DIGIT Depth Map", color_depth)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_depth()
