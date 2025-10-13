import cv2
from picamera2 import Picamera2
import time

# Initialize the camera
picam2 = Picamera2()

# Configure for video recording
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

# Let the camera warm up
time.sleep(2)

# Define video output settings
fps = 30  # frames per second
output_file = "video_output.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (640, 480))

print("🎥 Recording video... Press 'q' to stop.")

# Capture frames until 'q' is pressed
while True:
    frame = picam2.capture_array()
    cv2.imshow("Recording", frame)
    video_writer.write(frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("✅ Recording stopped. Video saved as:", output_file)

# Cleanup
video_writer.release()
cv2.destroyAllWindows()
picam2.stop()
picam2.close()
