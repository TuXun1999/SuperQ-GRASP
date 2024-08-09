import pyrealsense2 as rs
import numpy as np
import cv2


class CaptureArena():
	def __init__(self):
		# Initialization function
		# Just use the default setup, without additional change
		self.pipe = rs.pipeline()
		self.cfg = rs.config()

		self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
		self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

		self.pipe.start(self.cfg)


	# The function to visualize the result
	def vis(self):
		# Show the image
		while True:
			color_image = self.grab_rgb_image()
			cv2.imshow('rgb', color_image)

			if cv2.waitKey(1) == ord('q'):
				break


	# The function to capture an image and detect the apriltags in it
	def grab_rgb_image(self):
		frame = self.pipe.wait_for_frames()
		color_frame = frame.get_color_frame()

		# Convert the data into a numpy array
		color_image = np.asanyarray(color_frame.get_data())

		# Detect the AprilTag
		gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
		

		return color_image


	def close(self):
		self.pipe.stop()

def main():
	capture_area = CaptureArena()
	# Grab an image to obtain the transformation
	capture_area.vis()


	capture_area.close()
	


if __name__ == "__main__":
    main()