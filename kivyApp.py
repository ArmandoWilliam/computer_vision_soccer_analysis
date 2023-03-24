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
from kivy.graphics import Rectangle, Color

import main_app
from main_app import cv
import numpy as np

FPS = 30


class MyPopup(Popup):
    def __init__(self, **kwargs):
        super(MyPopup, self).__init__(**kwargs)
        self.title = 'GOAL'
        self.content = Label(text='GOOOOOOOOL')
        
class PercentageBar(BoxLayout):
    def __init__(self, first_percentage, second_percentage, **kwargs):
        super().__init__(**kwargs)
        
        # calculate the width of each rectangle based on the percentages
        first_percentage_width = first_percentage / 100 * self.width
        second_percentage_width = second_percentage / 100 * self.width
        
        # create the white rectangle
        with self.canvas:
            Color(1, 1, 1)
            self.white_rect = Rectangle(pos=self.pos, size=(first_percentage_width, self.height))
        
        # create the blue rectangle
        with self.canvas:
            Color(0, 0, 1)
            self.blue_rect = Rectangle(pos=(self.pos[0] + first_percentage_width, self.pos[1]), size=(second_percentage_width, self.height))
class SoccerApp(App):

    (H, W) = (None, None)
    writer = None
    interval = None
    
    def build(self):
        # Create a box layout with two buttons and an Image widget
        box_layout = BoxLayout(orientation='vertical', spacing=10, padding=20)
        
        # Create the Image widget and add it to the box layout
        self.image = Image(allow_stretch=True, keep_ratio=True, size_hint=(1, 1))
        box_layout.add_widget(self.image)

        # Add the nameVideo TextInput and Submit button in a horizontal box layout
        input_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        self.nameVideo = TextInput(text="videos/mc_possession.mp4", multiline=False, size_hint=(0.7, 1))
        input_box.add_widget(self.nameVideo)
        input_box_2 = BoxLayout(orientation='horizontal', size_hint=(0.3, 1))
        input_box_2.submit = Button(text="Submit", italic=True, size_hint=(0.5, 1))

        input_box_2.add_widget(input_box_2.submit)
        input_box.add_widget(input_box_2)
        box_layout.add_widget(input_box)

        # Create the buttons and place them in a horizontal box layout
        horizontal_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        btn_start = Button(text='Start Detection', italic=True, size_hint=(0.5, 1))
        btn_stop = Button(text='Stop Detection', italic=True, size_hint=(0.5, 1))
        horizontal_box.add_widget(btn_start)
        horizontal_box.add_widget(btn_stop)
        box_layout.add_widget(horizontal_box)

        # Bind the buttons to their respective functions
        input_box_2.submit.bind(on_press=self.upload_video)
        btn_start.bind(on_press=self.start_detection)
        btn_stop.bind(on_press=self.stop_detection)

        # Create the PercentageBar widget and add it to the box layout
        self.percentage_bar = PercentageBar(50, 50, size_hint=(1, 0.1))
        box_layout.add_widget(self.percentage_bar)

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
            
             # Update the percentage bar
            self.percentage_bar.white_rect.size = (percentage_possession_team_A / 100 * self.percentage_bar.width, self.percentage_bar.height)
            self.percentage_bar.blue_rect.pos = (self.percentage_bar.white_rect.pos[0] + self.percentage_bar.white_rect.size[0], self.percentage_bar.pos[1])
            self.percentage_bar.blue_rect.size = (percentage_possession_team_B / 100 * self.percentage_bar.width, self.percentage_bar.height)
             
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
                Clock.schedule_interval(self.update_image,1/FPS)


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