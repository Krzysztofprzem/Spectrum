import cv2
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq

def load_image(path="default.jpg"):
    """
    Method to load background image from path
    :param path:
    :return:
    """
    image = cv2.imread(path)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    return image

def load_samples(path):
    """
    Method to load samples from .wav file determined by path
    :param path:
    :return: sampling frequency, list of normalized channels
    """
    fs, channels = wavfile.read(path)
    channels = np.transpose(channels)/255
    return fs, channels


def compute_FFT(channels, N):
    """
    Method to compute FFT
    :param channels: list of channels
    :param N: size of FFT
    :return: list of magnitudes of channels
    """
    channels_magnitude = []
    for channel in channels:
        channel_fft = fft(channel[1000000:], N)
        magnitude = abs(channel_fft)
        channels_magnitude.append(np.ndarray.tolist(magnitude))
    return channels_magnitude



def get_coords_of_lines(center_x, center_y, r, phase, delta_fi, j, harmonic):
    """
    Method to determine start and end of line of magnitude
    :param center_x: x coord of center of image
    :param center_y: y coord of center of image
    :param r: radius of central circle
    :param phase: start phase
    :param delta_fi: shift of current sample
    :param j: number of sample from channel
    :param harmonic: magnitude of j'th sample
    :return: start and end points of line
    """
    point1 = [center_x+r*np.cos(phase+delta_fi*j), center_y-r*np.sin(phase+delta_fi*j)]
    point2 = [center_x+(r+harmonic)*np.cos(phase+delta_fi*j), center_y-(r+harmonic)*np.sin(phase+delta_fi*j)]
    for i in range(len(point1)):
        point1[i] = int(point1[i])
        point2[i] = int(point2[i])
    return tuple(point1), tuple(point2)



def draw_spectrum(frame, channels_magnitude, r, N):
    """
    Method to draw spectrum on circle on frame
    :param frame: frame
    :param channels_magnitude:
    :param r:
    :param N:
    :return:
    """
    h, w, _ = frame.shape
    center_y = int(h/2)
    center_x = int(w/2)

    phase_shift = 0

    for i in range(len(channels_magnitude)):
        phase = i*2*np.pi/len(channels_magnitude) + (np.pi/2) + phase_shift
        delta_fi = 2*np.pi/len(channels_magnitude)/N
        for j in range(len(channels_magnitude[i])):
            point1, point2 = get_coords_of_lines(center_x, center_y, r, phase, delta_fi, j, channels_magnitude[i][j])
            frame = cv2.line(frame, point1, point2, (0, 0, 255), 1)
            frame = cv2.circle(frame, point2, 2, (0, 0, 255), -1)
            cv2.imshow("frame", frame)
            cv2.waitKey(10)


def draw_circle(frame, channels_magnitude, r):
    h, w, _ = frame.shape
    center_y = int(h/2)
    center_x = int(w/2) 

    frame = cv2.circle(frame, (center_x, center_y), r, (0,0,255), 2)
    frame = cv2.blur(frame, (3, 3))
    # cv2.imshow("img", frame)
    # cv2.waitKey(0)
    return frame


def draw(frame, channels_magnitude, N):
    """
    Method responsible for calling all drawing methods
    :param frame: frame
    :param channels_magnitude:
    :param N: size of FFT
    :return: frame with drawed spectrum
    """
    r = 75
    draw_spectrum(frame, channels_magnitude, r, N)
    #draw_circle(frame, channels_magnitude, r)
    return frame

def main():
    framerate = 60      # frames per second
    N = 128

    image = load_image()
    fs, channels = load_samples("beelze.wav")

    channels_magnitude = compute_FFT(channels, N)

    # while True:
    frame = image.copy()
    frame = draw(frame, channels_magnitude, N)
    print(fs)
    print((channels_magnitude))






if __name__ == "__main__":
    main()