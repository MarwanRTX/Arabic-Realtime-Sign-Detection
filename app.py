import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from collections import deque, Counter
from gtts import gTTS
import os
import pygame
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display

# ====== Load Trained Model & Labels ======
font_path = "Amiri-Regular.ttf"
font_pil = ImageFont.truetype(font_path, 48)  # size can be changed

model = tf.keras.models.load_model("best_model.h5")
class_names = list(np.load("label_classes.npy", allow_pickle=True))
print("✅ Model & labels loaded!")

# ====== Map English Labels to Arabic Characters ======
label_to_arabic = {
    'Ain': 'ع',
    'Al': 'ال',
    'Alef': 'أ',
    'Beh': 'ب',
    'Dad': 'ض',
    'Dal': 'د',
    'Feh': 'ف',
    'Ghain': 'غ',
    'Hah': 'ح',
    'Heh': 'ه',
    'Jeem': 'ج',
    'Kaf': 'ك',
    'Khah': 'خ',
    'Laa': 'لا',
    'Lam': 'ل',
    'Meem': 'م',
    'Noon': 'ن',
    'Qaf': 'ق',
    'Reh': 'ر',
    'Sad': 'ص',
    'Seen': 'س',
    'Sheen': 'ش',
    'Tah': 'ط',
    'Teh': 'ت',
    'Teh_Marbuta': 'ة',
    'Thal': 'ذ',
    'Theh': 'ث',
    'Waw': 'و',
    'Yeh': 'ي',
    'Zah': 'ظ',
    'Zain': 'ز'
}


# ====== Mediapipe Setup ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ====== Webcam ======
cap = cv2.VideoCapture(0)

cooldown = 3
last_pred_time = time.time() - cooldown
prediction_queue = deque(maxlen=5)
CONFIDENCE_THRESHOLD = 0.6
typed_text = ""
last_letter = ""
last_speak_time = 0

# ====== Audio Init ======
pygame.mixer.init()

def speak_arabic(text):
    # Stop previous audio if still playing
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

    # Save new TTS file and play it
    tts = gTTS(text=text, lang='ar')
    import uuid
    filename = f"speech_{uuid.uuid4().hex}.mp3"
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    info_msg = f"Say letter every {cooldown}s | Press 'c' to clear | 'q' to quit"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            input_tensor = np.array(keypoints, dtype=np.float32).reshape(1, -1)

            if time.time() - last_pred_time >= cooldown:
                preds = model.predict(input_tensor, verbose=0)[0]
                conf = np.max(preds)
                pred_class = np.argmax(preds)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(frame_pil)
                draw.text((10, h - 80), typed_text, font=font_pil, fill=(255, 255, 0))


                if conf > CONFIDENCE_THRESHOLD:
                    label = class_names[pred_class]
                    arabic_letter = label_to_arabic.get(label, label)

                    # Avoid repeating same letter every time
                    if arabic_letter != last_letter:
                        typed_text += arabic_letter
                        speak_arabic(arabic_letter)
                        last_letter = arabic_letter
                        last_pred_time = time.time()

    # === Display Prediction & Full Typed Word ===
    # === Convert frame to PIL image to draw Arabic text ===
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(frame_pil)
    reshaped_text = arabic_reshaper.reshape(typed_text)
    bidi_text = get_display(reshaped_text)
    draw.text((10, h - 80), bidi_text, font=font_pil, fill=(153, 0, 0))

    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV


    cv2.putText(frame, info_msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (153, 0, 0), 2)

    cv2.imshow("✋ Arabic Sign Detector", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    elif key == ord("c"):
        typed_text = typed_text[:-1]  
        

cap.release()
cv2.destroyAllWindows()
