# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.checkbox import CheckBox
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.resources import resource_find
import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# Constants
IMG_SIZE = 640

class RootWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        # Display area
        self.img_widget = Image(size_hint=(1, 0.8))
        self.add_widget(self.img_widget)

        # Controls area
        ctrl = BoxLayout(size_hint=(1, 0.2), spacing=10)

        # Threshold sliders
        self.mouth_thresh = Slider(min=0, max=1, value=0.6)
        self.tooth_thresh = Slider(min=0, max=1, value=0.1)
        self.hue_shift   = Slider(min=0, max=179, value=0)
        self.white_check = CheckBox(active=False)

        # File action buttons
        btn_image  = Button(text='Load Image', on_release=self.show_filechooser_image)
        btn_video  = Button(text='Load Video', on_release=self.show_filechooser_video)
        btn_cam    = Button(text='Start Webcam', on_release=self.start_webcam)

        # Layout assembly
        ctrl.add_widget(btn_image)
        ctrl.add_widget(btn_video)
        ctrl.add_widget(btn_cam)
        ctrl.add_widget(self.mouth_thresh)
        ctrl.add_widget(self.tooth_thresh)
        ctrl.add_widget(self.hue_shift)
        ctrl.add_widget(self.white_check)
        self.add_widget(ctrl)

        # Load models
        self._setup_models()

        # VideoCapture placeholder
        self.capture = None
        self.source_type = None  # 'image','video','cam'
        self.video_capture = None

        # Schedule update only when streaming
        self._event = None

    def _setup_models(self):
        # Resolve .pt in APK assets or cwd
        mouth_path = resource_find('mouth_detect.pt') or os.path.join(os.getcwd(), 'mouth_detect.pt')
        seg_path   = resource_find('tooth_seg.pt')   or os.path.join(os.getcwd(), 'tooth_seg.pt')
        self.mouth_model = YOLO(mouth_path)
        self.seg_model   = YOLO(seg_path, task='segment')

    def show_filechooser_image(self, *args):
        self.source_type = 'image'
        self._open_filechooser('Select Image', self.load_image)

    def show_filechooser_video(self, *args):
        self.source_type = 'video'
        self._open_filechooser('Select Video', self.load_video)

    def _open_filechooser(self, title, callback):
        chooser = FileChooserIconView()
        popup = Popup(title=title, content=chooser, size_hint=(0.9, 0.9))
        chooser.bind(on_submit=lambda inst, selection, touch: (callback(selection[0]), popup.dismiss()))
        popup.open()

    def load_image(self, path):
        # Stop any streaming
        self._stop_stream()
        img = cv2.imread(path)
        out = self.safe_process(img)
        self._update_texture(out)

    def load_video(self, path):
        self.source_type = 'video'
        self._stop_stream()
        self.video_capture = cv2.VideoCapture(path)
        self._event = Clock.schedule_interval(self._video_step, 1/30.)

    def start_webcam(self, *args):
        self.source_type = 'cam'
        self._stop_stream()
        self.video_capture = cv2.VideoCapture(0)
        self._event = Clock.schedule_interval(self._video_step, 1/30.)

    def _video_step(self, dt):
        ret, frame = self.video_capture.read()
        if not ret:
            self._stop_stream()
            return
        out = self.safe_process(frame)
        self._update_texture(out)

    def _stop_stream(self):
        if self._event:
            self._event.cancel()
            self._event = None
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

    def safe_process(self, frame):
        try:
            return self.process_frame(frame)
        except Exception as e:
            print(f"Processing error: {e}")
            return frame

    def process_frame(self, frame):
        # Mouth detection
        res = self.mouth_model.predict(source=frame, imgsz=IMG_SIZE)[0]
        mouth_box = None
        for bbox, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            if conf >= self.mouth_thresh.value and res.names[int(cls)].lower()=='mouth':
                x1,y1,x2,y2 = map(int, bbox.tolist())
                mouth_box = (x1,y1,x2,y2)
                break
        if not mouth_box:
            return frame

        # Crop bounds
        x1,y1,x2,y2 = mouth_box
        h,w = frame.shape[:2]
        x1, x2 = np.clip([x1,x2], 0, w)
        y1, y2 = np.clip([y1,y2], 0, h)
        crop = frame[y1:y2, x1:x2]

        # Segmentation
        seg = self.seg_model.predict(source=crop, imgsz=IMG_SIZE)[0]
        masks = []
        if seg.masks is not None: # Add this check
            for m, conf in zip(seg.masks.data.cpu().numpy(), seg.boxes.conf.cpu().numpy()):
                if conf >= self.tooth_thresh.value:
                    masks.append(cv2.resize(m, (x2-x1, y2-y1)))

        if masks:
            frame = self._apply_hue(frame, masks, (y1, x1))
        return frame

    def _apply_hue(self, frame, masks, offset):
        y_off, x_off = offset
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        mask_map = np.zeros_like(h, bool)
        for m in masks:
            region = (m > 0.5)
            mask_map[y_off:y_off+m.shape[0], x_off:x_off+m.shape[1]] |= region
        if self.white_check.active:
            s[mask_map] = 0
        else:
            h[mask_map] = (h[mask_map] + int(self.hue_shift.value)) % 180
        return cv2.cvtColor(cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)

    def _update_texture(self, frame):
        if frame is None or frame.size == 0:
            return
        buf = cv2.flip(frame, 0).tobytes()
        tex = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        tex.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img_widget.texture = tex

class ToothApp(App):
  use_kv = False
  def build(self):
    return RootWidget()

if __name__ == '__main__':
    ToothApp().run()