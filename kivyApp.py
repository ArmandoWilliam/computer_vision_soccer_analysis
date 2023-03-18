from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.uix.modalview import ModalView
import time
import os.path

import main_app
from main_app import cv
import numpy as np


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
        box_layout = BoxLayout(orientation='vertical',spacing=10,padding=20)
        horizontal_box=BoxLayout(orientation='horizontal')
        input_box=BoxLayout(orientation='horizontal')
        input_box_2=BoxLayout(orientation='horizontal')
        # Add a label for the buttons
        # label=Label(text="Name (add the extension):",size_hint=(0.1,1))
        # input_box.add_widget(label)
        # self.nameVideo=TextInput(text="example: soccer.mp4",multiline=True)
        self.nameVideo=TextInput(text="videos/goals_2.mp4",multiline=False, size_hint=(0.3,0.5))
        input_box_2.add_widget(self.nameVideo)
        input_box_2.submit=Button(text="Submit",italic=True,size_hint=(0.3,0.5))
        input_box_2.add_widget(input_box_2.submit)

        box_layout.add_widget(input_box)
        box_layout.add_widget(input_box_2)

        # Create the buttons
        btn_start = Button(text='Start Detection',italic=True,size_hint=(0.3,0.5))
        btn_stop = Button(text='Stop Detection',italic=True,size_hint=(0.3,0.5))

        # Bind the buttons to their respective functions
        input_box_2.submit.bind(on_press=self.upload_video)
        btn_start.bind(on_press=self.start_detection)
        btn_stop.bind(on_press=self.stop_detection)

        
        # Create the Image widget and add it to the box layout
        self.image = Image(allow_stretch=True, keep_ratio=True, size_hint=(1, 1), size=(box_layout.width, box_layout.height))
        box_layout.add_widget(self.image)

        horizontal_box.add_widget(btn_start)
        horizontal_box.add_widget(btn_stop)
        box_layout.add_widget(horizontal_box)

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
        self.frame = None
        # Call the Clock schedule_interval method to update the image
        self.interval = Clock.schedule_interval(self.update_image,1/25)
    
    # to keep the past frame
    # prev_frame = None

    def update_image(self, dt):
        # global prev_frame
        ret, frame = self.capture.read()
        if ret:
            #Create the writer to write the file, it should execute only the first time I press the start button
            if self.writer is None: 
                (H, W) = frame.shape[:2]
                self.fourcc = cv.VideoWriter_fourcc(*"MJPG")
                self.writer = cv.VideoWriter("result.avi", self.fourcc, 25, (W, H), True)

            # Perform detection with YOLO and tracking with algorithms
            # Detect the colors of the players and the ball
            edit_image, goal_result = main_app.process_one_image(frame)
            # prev_frame = edit_image.copy().astype(np.float32)

            paused = False

            def on_popup_dismiss(popup):
                global paused
                paused = True
                Clock.unschedule()
                print('Popup dismissed')
            
            def on_play_button_press(button):
                global paused
                paused = False
                Clock.schedule_interval(self.update_image,1/25)


            #save the frame in the video "result.avi"
            """ if goal_result is not None and goal_result > 0:
                # Pause for 2 seconds
                time.sleep(2)
                # Create the popup
                popup = MyPopup(size_hint=(None, None), size=(400, 400))
                # popup.bind(on_dismiss=on_popup_dismiss)
                # Display the popup
                popup.open()
                # Pause execution until the popup is dismissed
                # popup.wait_for_dismiss() """

            self.writer.write(edit_image)

            frame_BGR = cv.cvtColor(edit_image, cv.COLOR_RGB2BGR)
            # Convert the processed frame to an image texture
            buf1 = cv.flip(frame_BGR, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(frame_BGR.shape[1], frame_BGR.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

            # Update the Image widget with the new image texture
            self.image.texture = texture


    def stop_detection(self, instance):
        # Call the Clock unschedule method to stop the image updates
        if self.interval:
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