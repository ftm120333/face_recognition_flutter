# test_face_reco

# 👁️‍🗨️ Face Detection & Recognition App (Flutter + ML Kit + TensorFlow Lite)

A real-time face recognition mobile app built with **Flutter**, leveraging **Google ML Kit** for face detection and a custom **TFLite MobileFaceNet** model for face recognition. Designed for scalable identity verification, this app compares live camera frames with a stored face embedding database.

---

## 📱 Features

- 📷 Real-time **Face Detection** using Google ML Kit
- 🤖 **Face Recognition** with MobileFaceNet (TensorFlow Lite)
- 🔒 Embedding-based identity matching (3 embeddings per person)
- 🧠 lightweight face embeddings (128D)
- ✏️ Prompts user for integer input after identification (custom workflow)
- 🚀 Flutter-optimized for smooth camera and ML processing

---

## 🧰 Tech Stack

| Layer              | Tool / Library                     |
|--------------------|------------------------------------|
| Frontend           | Flutter                            |
| Face Detection     | Google ML Kit                      |
| Face Recognition   | TensorFlow Lite (MobileFaceNet)    |
| Embedding Matching | Euclidean Distance / Cosine Similarity |
| Camera Feed        | Flutter camera plugin              |

---

## 🧪 How It Works

1. The app opens the **camera preview** in real time.
2. **Google ML Kit** detects one or more faces in each frame.
3. Each face is **aligned and cropped**, then passed to the **TFLite model**.
4. The model generates a **128-dimension embedding** for each face.
5. The app sends the embedding to an external API which:
   - Compares it against stored embeddings (3 per person).
   - Returns the **matched identity (or unknown)**.
6. The app prompts the user for an **integer input** based on the result.
7. That input, along with identity data, is sent to the backend.

---
