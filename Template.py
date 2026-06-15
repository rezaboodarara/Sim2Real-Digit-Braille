# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import cv2
import hydra
import pybullet as p
import pybulletX as px
import tacto  # Import TACTO
import numpy as np
from PIL import Image


log = logging.getLogger(__name__)

def depth_to_normals(depth_map):
    zy, zx = np.gradient(depth_map)
    normal = np.dstack((-zx, -zy, np.ones_like(depth_map)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    
    normal_map_uint8 = ((normal + 1) /2 *255).astype(np.uint8)
    
    return cv2.cvtColor(normal_map_uint8, cv2.COLOR_RGB2BGR)

# Load the config YAML file from examples/conf/digit.yaml
@hydra.main(config_path="conf", config_name="Template")
def main(cfg):
    # Initialize digits
    bg = cv2.imread("conf/bg_digit_240_320.jpg")
    digits = tacto.Sensor(**cfg.tacto, background=bg)

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Create and initialize DIGIT
    digit_body = px.Body(**cfg.digit)
    digits.add_camera(digit_body.id, [-1])

    # Add object to pybullet and tacto simulator
    obj = px.Body(**cfg.object)
    digits.add_body(obj)

    # Create control panel to control the 6DoF pose of the object
    panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel)
    panel.start()
    log.info("Use the slides to move the object until in contact with the DIGIT")

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    counter = 0
    while True:
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        
        #at least 3 loop is needed to waste time and let the sensor capture warm up and be loaded.
        if counter <= 2:
            counter +=1
            
            image_rgb = color[0]
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite("tactile_output.png", image_bgr)
            
            current_depth = depth[0]
            normal_img = depth_to_normals(current_depth)
            depth_visual = cv2.normalize(current_depth, None, 0, 255,cv2.NORM_MINMAX)
            cv2.imwrite("depth_output.png", depth_visual)
            
        if counter == 2:
            break
    t.stop()


if __name__ == "__main__":
    main()
