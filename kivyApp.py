from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.uix.modalview import ModalView
from kivy.uix.widget import Widget
import time
import os.path
from kivy.graphics import Rectangle, Color, RoundedRectangle
from kivy.properties import NumericProperty

import main_app
from main_app import cv
import numpy as np

FPS = 30


class MyPopup(Popup):
    def __init__(self, **kwargs):
        super(MyPopup, self).__init__(**kwargs)
        self.title = 'GOAL'
        self.content = Label(text='GOOOOOOOOL')

class SoccerApp(App):

    (H, W) = (None, None)
    writer = None
    interval = None
    
    def build(self):
        # Create a box layout with two buttons and an Image widget
        box_layout = BoxLayout(orientation='vertical', spacing=10, padding=20)
        
        # Create the Image widget and add it to the box layout
        self.image = Image(allow_stretch=True, keep_ratio=True, size_hint=(1, 1), size=(box_layout.width, box_layout.height))
        box_layout.add_widget(self.image)
        
        # Create the nameVideo TextInput and Submit button and add them to a horizontal box layout
        input_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=50, spacing=10)
        self.nameVideo = TextInput(text="videos/mc_possession.mp4", multiline=False, size_hint=(0.7, 1))
        input_box.add_widget(self.nameVideo)
        self.input_box_2 = BoxLayout(orientation='horizontal', size_hint=(0.3, 1), spacing=10)
        self.input_box_2.submit = Button(text="Submit", italic=True, size_hint=(0.5, 1))
        self.input_box_2.add_widget(self.input_box_2.submit)
        input_box.add_widget(self.input_box_2)
        
        # Add the nameVideo and Submit box to the main box layout
        box_layout.add_widget(input_box)
        
        # Create the buttons and add them to a horizontal box layout
        horizontal_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=50, spacing=10)
        self.btn_start = Button(text='Start Detection', italic=True, size_hint=(0.5, 1))
        self.btn_stop = Button(text='Stop Detection', disabled= True, italic=True, size_hint=(0.5, 1))
        horizontal_box.add_widget(self.btn_start)
        horizontal_box.add_widget(self.btn_stop)
        
        # Bind the buttons to their respective functions
        self.input_box_2.submit.bind(on_press=self.upload_video)
        self.btn_start.bind(on_press=self.start_detection)
        self.btn_stop.bind(on_press=self.stop_detection)
        
        # Add the buttons to the main box layout
        box_layout.add_widget(horizontal_box)
        
        # Create the grid layout for displaying the percentage values
        percentage_layout = GridLayout(cols=2, spacing=10, size_hint=(1, None), height=50)
        percentage_layout.add_widget(Label(text='Team A'))
        self.percentage_label_a = Label(text='0%')
        percentage_layout.add_widget(self.percentage_label_a)
        percentage_layout.add_widget(Label(text='Team B'))
        self.percentage_label_b = Label(text='0%')
        percentage_layout.add_widget(self.percentage_label_b)

        # Add the percentage layout to the main layout
        box_layout.add_widget(percentage_layout)
        
        return box_layout


    
    def upload_video(self,instance):
        self.titoloVideo=self.nameVideo.text
        if(os.path.isfile(self.titoloVideo)):
            box_popup_video_added = BoxLayout(orientation = 'vertical', padding = (100))
            box_popup_video_added.add_widget(Label(text="video uploaded successfully"))
            btn1 = Button(text = "Close")
            box_popup_video_added.add_widget(btn1)
            pop=Popup(title="Check video",content=box_popup_video_added,auto_dismiss=False)
            btn1.bind(on_press = pop.dismiss)
            pop.open()
        else:
            box_popup2 = BoxLayout(orientation = 'vertical', padding = (100))
            box_popup2.add_widget(Label(text="error: video not found!"))
            btn2 = Button(text = "Close")
            box_popup2.add_widget(btn2)
            pop=Popup(title="Check video",content=box_popup2,auto_dismiss=False)
            btn2.bind(on_press = pop.dismiss)
            pop.open()

    def start_detection(self, instance):
        self.capture = cv.VideoCapture(self.titoloVideo)
        self.btn_start.disabled = True
        self.btn_stop.disabled = False
        self.input_box_2.submit.disabled = True
        # Call the Clock schedule_interval method to update the image
        self.interval = Clock.schedule_interval(self.update_image,1/FPS)
        
        
    def update_percentage_bar(self, current_size, total_size):
        percentage = (current_size / total_size) * 100
        self.percentage_text.text = f"{percentage:.2f}%"
        self.blue_rect.size = (self.percentage_bar.width * (percentage / 100), self.blue_rect.height)
    
    def update_image(self, dt):
        # global prev_frame
        ret, frame = self.capture.read()
        if ret:
            #Create the writer to write the file, it should execute only the first time I press the start button
            if self.writer is None: 
                (H, W) = frame.shape[:2]
                self.fourcc = cv.VideoWriter_fourcc(*"MJPG")
                self.writer = cv.VideoWriter("result.avi", self.fourcc, FPS, (W, H), True)

            # Perform detection with YOLO and tracking with algorithms
            # Detect the colors of the players and the ball
            edit_image, percentage_possession_team_A, percentage_possession_team_B = main_app.process_one_image(frame, FPS)
            
            self.percentage_label_a.text = f"{int(round(percentage_possession_team_A,0))}%"
            self.percentage_label_b.text = f"{int(round(percentage_possession_team_B,0))}%"
             
            # prev_frame = edit_image.copy().astype(np.float32)

            #save the frame in the video "result.avi"

            self.writer.write(edit_image)

            frame_BGR = cv.cvtColor(edit_image, cv.COLOR_RGB2BGR)
            # Convert the processed frame to an image texture
            buf1 = cv.flip(frame_BGR, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(frame_BGR.shape[1], frame_BGR.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

            # Update the Image widget with the new image texture
            self.image.texture = texture
        else:
            if self.interval:
                self.capture.release()
                Clock.unschedule(self.interval)
            
            if os.path.isfile("result.avi"):
                box_popup_detection_stopped = BoxLayout(orientation = 'vertical', padding = (100))
                box_popup_detection_stopped.add_widget(Label(text="detection stopped, video saved as result.avi"))
                btn1 = Button(text = "Close")
                box_popup_detection_stopped.add_widget(btn1)
                pop=Popup(title="Detection Stopped",content=box_popup_detection_stopped,auto_dismiss=False)
                btn1.bind(on_press = pop.dismiss)
                pop.open()
                # Update the Image widget 
                self.image.texture = None
            return


    def stop_detection(self, instance):
        # Call the Clock unschedule method to stop the image updates
        if self.interval:
            self.capture.release()
            Clock.unschedule(self.interval)
        if os.path.isfile("result.avi"):
            box_popup_detection_stopped = BoxLayout(orientation = 'vertical', padding = (100))
            box_popup_detection_stopped.add_widget(Label(text="detection stopped, video saved as result.avi"))
            btn1 = Button(text = "Close")
            box_popup_detection_stopped.add_widget(btn1)
            pop=Popup(title="Detection Stopped",content=box_popup_detection_stopped,auto_dismiss=False)
            btn1.bind(on_press = pop.dismiss)
            pop.open()
            # Update the Image widget 
            self.image.texture = None
        
        self.btn_start.disabled = False
        self.btn_stop.disabled = True
        self.input_box_2.submit.disabled = False