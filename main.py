import face_recognition
import cv2
#
import smtplib
import time
from email.message import EmailMessage

# Load a known image and encode it
known_image1 = face_recognition.load_image_file("Riwaz.jpg")
known_image2 = face_recognition.load_image_file("Kushal.jpg")
known_image3 = face_recognition.load_image_file("Liam.jpg")
known_encoding1 = face_recognition.face_encodings(known_image1)[0]
known_encoding2 = face_recognition.face_encodings(known_image2)[0]
known_encoding3 = face_recognition.face_encodings(known_image3)[0]
known_encodings = [known_encoding1, known_encoding2, known_encoding3]
known_names = ["Riwaz","Kushal", "Liam"]

#

# Track last email sent time
last_sent = {}

    # Email function
def send_email(name):
    msg = EmailMessage()
    msg.set_content(f"{name} was detected by the system.")

    msg["Subject"] = f"Face Recognized: {name}"
    msg["From"] = "riwazshrestha2005@gmail.com"
    msg["To"] = "riwazshrestha2005@gmail.com"

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login("riwazshrestha2005@gmail.com", "fubn ixmb jbfw gtnz")
        smtp.send_message(msg)
        print(f"Email sent for {name}")


#

sent_time=0
timeout = 50

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    #
    if not ret:
        break
    #

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all faces and encode them
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    similarity = 0
    name="Unknown"

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known face
        matches = face_recognition.compare_faces(known_encodings, face_encoding) # list of boolean values true/false
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin()

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin()

        if matches[best_match_index]:
            name = known_names[best_match_index]

            # Calculate similarity percentage
            similarity = (1 - face_distances[best_match_index]) * 100

            if similarity >= 50:
                # Cooldown: Only send email once every 5 minutes per person
                current_time = time.time()
                if name in known_names and (current_time - sent_time > timeout):
                    send_email(name)
                    sent_time = current_time

        # Scale 
        # face locations back to original size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Draw similarity index under the name
        cv2.putText(frame, f"{similarity:.2f}%", (left, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)


    # Show video
    cv2.imshow('Face Recognition', frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
