from digit_interface import Digit
import cv2
import os
import time
from PIL import Image

serial_number = "D20995"
save_dir = "Test"
letter = "7"
capture_interval = 2 #seconds
num_frames = 100 #number of needed samples

os.makedirs(save_dir, exist_ok=True)

d = Digit(serial_number)
d.connect()
time.sleep(2)
# d.show_view()

#For some unkonwn reason it need some get frame with some dalayed intervals to get calibrated and produce write photos
for i in range(5):
    frame = d.get_frame() #warmup
    time.sleep(1)

try:
    for i in range(20,num_frames+20):

        #Again for some unknown reason it needs some warm up
        print("warmup...")
        for j in range(5):
            frame = d.get_frame() #warmup
            time.sleep(1)
        frame = d.get_frame()
        
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = os.path.join(save_dir, f"{letter}-{i}.png")
        print(filename)
        # cv2.imwrite(filename, frame_bgr)
        image = Image.fromarray(frame)
        image.save(filename)
        print(f"{i}/{num_frames+20} Saved frame as {filename}")
        if i < num_frames - 1:
            time.sleep(capture_interval)
finally:
    d.disconnect()
    print("Disconnected from Digit.")



# d = Digit("D20995") # Unique serial number
# d.connect()
# d.show_view()
# d.disconnect()
