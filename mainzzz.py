import os
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import datetime
import util

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)
        
        self.db_dir = 'db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
    
        # log data of all users
        self.log_path = './log.txt'
    
    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            # to open camera once not calll obj again and agin
            self.cap = cv2.VideoCapture(0)   
            
        self._label = label
        self.process_webcam()
        
    def process_webcam(self):
        # read frames from webcam and put these frames into labels
        # convert all frames and imgs into pillow to display
        ret, frame = self.cap.read()
        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil) 
        
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)
        # we are calling this same process again every 20ms
        self._label.after(10, self.process_webcam)  
                        
    def login(self):
        try:
            # Load known faces and encodings from the database
            known_faces_encodings = []
            known_names = []
            for file_name in os.listdir(self.db_dir):
                if file_name.endswith('.jpg'):
                    name = os.path.splitext(file_name)[0]
                    image_path = os.path.join(self.db_dir, file_name)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    # Assuming there's at least one face in each image
                    for encoding in encodings:
                        known_faces_encodings.append(encoding)
                        known_names.append(name)


            # Recognize the person in the most recent capture
            face_locations = face_recognition.face_locations(self.most_recent_capture_arr)
            if len(face_locations) == 0:
                util.msg_box('Ups...', 'No face found in the captured image. Please try again.')
                return

            # Extract face encodings from the most recent capture
            capture_encoding = face_recognition.face_encodings(self.most_recent_capture_arr, face_locations)[0]

            # Compare face encodings with known faces
            for known_encoding, name in zip(known_faces_encodings, known_names):
                match = face_recognition.compare_faces([known_encoding], capture_encoding)
                if match[0]:
                    util.msg_box('Welcome back!', f'Welcome, {name}.')
                    with open(self.log_path, 'a') as f:
                        f.write(f'{name},{datetime.datetime.now()},in\n')
                    return

            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        except Exception as e:
            util.msg_box('Error', f'An error occurred: {str(e)}')

    def logout(self):
        pass 
    
    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)
        
        #coz its not going to be a webcam
        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        # we want to add single img not webcam
        self.add_img_to_label(self.capture_label)
        
        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'WHO , \nARE YOU????')
        self.text_label_register_new_user.place(x=750, y=70)

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        
        ##capute picture
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        #everytime use clicks accept we have to take the name user inputs
        name = self.entry_text_register_new_user.get(1.0, "end-1c")
        
        # save image in the label / window 
        
        cv2.imwrite( os.path.join(self.db_dir , '{}.jpg'.format(name)) , self.register_new_user_capture )
        
        
        util.msg_box('Success!', f'{name} you are registered successfully !')
        self.register_new_user_window.destroy()
    
    def try_again_register_new_user(self):
        # exit and get back to main window
        self.register_new_user_window.destroy()
    
    def start(self):
        self.main_window.mainloop()
    
if __name__ == "__main__":
    app = App()
    app.start()
