import os.path
import datetime
import os
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import face_recognition
import util
import csv

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

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        # log data of all users
        self.log_path = './log.txt'
        # CSV file for attendance
        self.attendance_csv_path = './attendance.csv'

        # Load face encodings and names
        self.list_encodings, self.list_names = self.get_encoding()

        # Initialize attendance CSV
        self.initialize_attendance_csv()

        self.logged_in_user = None

    def initialize_attendance_csv(self):
        with open(self.attendance_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Name', 'Attendance']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for name in self.list_names:
                writer.writerow({'Name': name, 'Attendance': 'A'})

    def get_encoding(self):
        paths = [os.path.join(self.db_dir, f) for f in os.listdir(self.db_dir) if f.endswith('.jpg')]
        print('{} images found for encoding'.format(len(paths)))

        list_encodings = []
        list_names = []

        for img_path in paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(img_path)
            (name, ext) = os.path.splitext(basename)
            face_roi = face_recognition.face_locations(img, model='cnn')
            face_encoding = face_recognition.face_encodings(img, face_roi)[0]
            if len(face_encoding) > 0:
                list_encodings.append(face_encoding)
                list_names.append(name)
            else:
                print('Could not detect the face from image ', img_path)

        return list_encodings, list_names

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            # to open camera once not call obj again and again
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        # read frames from webcam and put these frames into labels
        # convert all frames and imgs into pillow to display
        ret, frame = self.cap.read()
        self.most_recent_capture_arr = frame

        # Corrected line: Convert to RGB before displaying
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)

        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)

        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        # we are calling this same process again every 20ms
        self._label.after(10, self.process_webcam)

    def login(self):
        # go to the database and check for all the images and recognize
        face_locations, face_names, conf_values = self.recognize_faces(self.most_recent_capture_arr)

        if len(face_names) > 0:
            # Person detected
            self.logged_in_user = face_names[0]
            util.msg_box('Welcome!', f'Welcome, {self.logged_in_user}!')
            with open(self.log_path, 'a') as f:
                f.write('{},{},in\n'.format(self.logged_in_user, datetime.datetime.now()))
                f.close()

            # Update attendance CSV
            self.update_attendance_csv(self.logged_in_user, 'P')
        else:
            # No person detected or not recognized
            util.msg_box('Sorry!', 'You are not registered. Please register or try again.')

    def recognize_faces(self, image, tolerance=0.6):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        face_names = []
        conf_values = []

        for encoding in face_encodings:
            matches = face_recognition.compare_faces(self.list_encodings, encoding, tolerance=tolerance)
            name = 'Not identified'
            face_distances = face_recognition.face_distance(self.list_encodings, encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.list_names[best_match_index]
            face_names.append(name)
            conf_values.append(face_distances[best_match_index])

        face_locations = np.array(face_locations)
        return face_locations.astype(int), face_names, conf_values

    def logout(self):
        if self.logged_in_user is not None:
            # Person logged in
            face_locations, face_names, conf_values = self.recognize_faces(self.most_recent_capture_arr)

            if len(face_names) > 0 and face_names[0] == self.logged_in_user:
                # Recognized and matched the logged-in user
                name = self.logged_in_user
                util.msg_box('Logged Out', f'Logged out of {name}.')
                with open(self.log_path, 'a') as f:
                    f.write('{},{},out\n'.format(name, datetime.datetime.now()))
                    f.close()

                # Update attendance CSV
                self.update_attendance_csv(name, 'P')
                self.logged_in_user = None
            elif len(face_names) > 0:
                # Recognized, but not the logged-in user
                util.msg_box('Sorry!', f'You are not {self.logged_in_user}. Cannot logout.')
            else:
                # No person detected or not recognized
                util.msg_box('Sorry!', 'Face not recognized. Cannot logout.')
        else:
            # No person logged in
            util.msg_box('Sorry!', 'You are not logged in. Cannot logout.')

    def update_attendance_csv(self, name, status):
        # Update attendance status in CSV
        with open(self.attendance_csv_path, 'r', newline='') as csvfile:
            fieldnames = ['Name', 'Attendance']
            reader = csv.DictReader(csvfile, fieldnames=fieldnames)
            rows = list(reader)

        for row in rows:
            if row['Name'] == name:
                row['Attendance'] = status

        with open(self.attendance_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green',
                                                                      self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again',
                                                                         'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        # coz its not going to be a webcam
        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        # we want to add a single img not webcam
        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'WHO , \nARE YOU????')
        self.text_label_register_new_user.place(x=750, y=70)

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        ##capture picture
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        # every time user clicks accept we have to take the name the user inputs
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        # save the image in the label / window
        cv2.imwrite(os.path.join(self.db_dir, '{}.jpg'.format(name)), self.register_new_user_capture)

        util.msg_box('Success!', f'{name} you are registered successfully !')
        self.register_new_user_window.destroy()

    def try_again_register_new_user(self):
        # exit and get back to the main window
        self.register_new_user_window.destroy()

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()
