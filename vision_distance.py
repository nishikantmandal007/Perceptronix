
import cv2
import numpy as np
import winsound


class vision_depth:
    def __init__(self):
        # Loading Mask RCNN
        self.net = cv2.dnn.readNetFromTensorflow("MODELS01/visdepth/frozen_inference_graph_coco.pb",
                                            "MODELS01/visdepth/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Generate random colors
        np.random.seed(2)
        self.colors = np.random.randint(0, 255, (90, 3))

        # Conf threshold
        self.detection_threshold = 0.7
        self.mask_threshold = 0.3

        self.classes = []
        with open("MODELS01/visdepth/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []

        # Distances
        self.distances = []

    def generate_sound(duration, frequency):
     volume = 0.5            # Range [0.0, 1.0]
     fs = 44100              # Sampling frequency (Hz)
     samples = duration * fs # Number of samples
     t = np.linspace(0, duration, samples, False) # Time axis

     # Generate the waveform
     note = volume * np.sin(2 * np.pi * frequency * t)

     # Convert to 16-bit format
     note = np.array(note * 32767, dtype=np.int16)

    #  # Play the sound
    #  p = pyaudio.PyAudio()
    #  stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, output=True)
    #  stream.write(note.tobytes())
    #  stream.stop_stream()
    #  stream.close()
    #  p.terminate()

    def detect_objects_mask(self, img):
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        self.net.setInput(blob)

        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])

        # Detect objects
        frame_height, frame_width, _ = img.shape
        detection_count = boxes.shape[2]

        # Object Boxes
        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []

        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            ''' if class_id=="bottle":
             generate_sound(0.5, 440) # Play a 440 Hz tone for 0.5 seconds'''


            score = box[2]
            color = self.colors[int(class_id)]
            if score < self.detection_threshold and class_id!="bottle":
                continue

            # Get box Coordinates
            x = int(box[3] * frame_width)
            y = int(box[4] * frame_height)
            x2 = int(box[5] * frame_width)
            y2 = int(box[6] * frame_height)
            self.obj_boxes.append([x, y, x2, y2])

            cx = (x + x2) // 2
            cy = (y + y2) // 2
            self.obj_centers.append((cx, cy))

            # append class
            self.obj_classes.append(class_id)

            # Contours
            # Get mask coordinates
            # Get the mask
            mask = masks[i, int(class_id)]
            roi_height, roi_width = y2 - y, x2 - x
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, self.mask_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.obj_contours.append(contours)

        return self.obj_boxes, self.obj_classes, self.obj_contours, self.obj_centers
    
   
    def draw_object_mask(self, img):
        # loop through the detection
        for box, class_id, contours in zip(self.obj_boxes, self.obj_classes, self.obj_contours):
            x, y, x2, y2 = box
            roi = img[y: y2, x: x2]
            
            roi_height, roi_width, _ = roi.shape
            color = self.colors[int(class_id)]

            roi_copy = np.zeros_like(roi)

            for cnt in contours:
               # # cv2.f(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                #cv2.drawContours(roi, [cnt], - 1, (int(color[0]), int(color[1]), int(color[2])), 3)
               # cv2.fillPoly(roi_copy, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                roi = cv2.addWeighted(roi, 1, roi_copy, 0.5, 0.0)
                img[y: y2, x: x2] = roi
        return img

    def draw_object_info(self, img, depth_map):
        # loop through the detection
        for box, class_id, obj_center,contours in zip(self.obj_boxes, self.obj_classes, self.obj_centers,self.obj_contours):
            x, y, x2, y2 = box

            color = self.colors[int(class_id)]
            color = (int(color[0]), int(color[1]), int(color[2]))

            cx, cy = obj_center

            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            x_dist = (cX - cx) * 0.001  # 0.001 is the pixel size, 24 is the focal length in mm, 120 is the baseline
            y_dist = (cY - cy) * 0.001
            z_dist = (24 * 120)/ depth_map[cY, cX]
            depth = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)

            #cv2.line(img, (cx, y), (cx, y2), color, 1)
            #cv2.line(img, (x, cy), (x2, cy), color, 1)
             
            class_name = self.classes[int(class_id)]
            #if class_name=="bottle":
             #cv2.rectangle(img, (x, y), (x + 250, y + 70), color, -1)
            cv2.putText(img, class_name.capitalize(), (x + 5, y + 25), 0, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "{} cm ".format(depth[0]), (x + 5, y + 60), 0, 1.0, (255, 255, 255), 2)
            
           
# ----------------------------------------------- SOUND --------------------------------------------------
            if str(class_name).__eq__('truck') and int(depth[0]) > 10:      # if depth is less than 25 cm then beep will sound.
                print(class_name)
                winsound.Beep(800, 2000)
            
             #cv2.rectangle(img, (x, y), (x2, y2), color, 1)




        return img





