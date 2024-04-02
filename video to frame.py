import cv2
import os

def is_blurry(image, threshold=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def extract_frames(video_path, output_folder, frames_per_second, blur_threshold=20):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if the video file exists
    if not os.path.exists(video_path):
        print("Video file does not exist.")
        return

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # Get frames per second

    try:
        # Iterate through video frames until the end
        while success:
            if count % int(fps / frames_per_second) == 0:
                # Check if the frame is too blurry
                if not is_blurry(image, blur_threshold):
                    # Write the frame image to file
                    cv2.imwrite(f"{output_folder}/frame{count:04d}.jpg", image)
            success, image = vidcap.read()
            count += 1
    except Exception as e:
        print("An error occurred:", e)
    finally:
        # Release the video object
        vidcap.release()
  
# Example usage
video_path = "birthday.mp4"
output_folder = "input images"
frames_per_second = 4 # Number of frames to capture per second
extract_frames(video_path, output_folder, frames_per_second)

