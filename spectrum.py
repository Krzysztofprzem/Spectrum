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
    channels = np.transpose(channels)/256
    return fs, channels


def compute_FFT(channels, N):
    """
    Method to compute FFT
    :param channels: list of channels
    :param N: size of FFT
    :return: list of magnitudes of channels
    """
    channels_magnitude = []
    scaler = N/16
    for channel in channels:
        channel_fft = fft(channel, N)
        magnitude = abs(channel_fft)
        channels_magnitude.append(np.ndarray.tolist(magnitude[:int(N/2)]/scaler))
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



def draw_spectrum(frame, channels_magnitude, r, N, phase_shift):
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

    delta_fi = 2*np.pi/len(channels_magnitude)/N
    for i in range(len(channels_magnitude)):
        phase = i*2*np.pi/len(channels_magnitude) + (np.pi/2) + phase_shift
        for j in range(len(channels_magnitude[i])):
            point1, point2 = get_coords_of_lines(center_x, center_y, r, phase, delta_fi, j, channels_magnitude[i][j])
            frame = cv2.line(frame, point1, point2, (71, 99, 255), 1)
            frame = cv2.circle(frame, point2, 2, (0, 0, 255), -1)
            # cv2.imshow("frame", frame)
            # cv2.waitKey(10)


def draw_circle(frame, channels_magnitude, r):
    h, w, _ = frame.shape
    center_y = int(h/2)
    center_x = int(w/2) 

    frame = cv2.circle(frame, (center_x, center_y), r, (0,0,255), 2)
    frame = cv2.blur(frame, (3, 3))
    # cv2.imshow("img", frame)
    # cv2.waitKey(0)
    return frame


def draw(frame, channels_magnitude, N, phase_shift):
    """
    Method responsible for calling all drawing methods
    :param frame: frame
    :param channels_magnitude:
    :param N: size of FFT
    :return: frame with drawed spectrum
    """
    r = 150
    draw_spectrum(frame, channels_magnitude, r, N, phase_shift)
    #draw_circle(frame, channels_magnitude, r)
    return frame

def main():
    # test = [0,1,2,3,4,5,6,7]
    # print(test[7-2:7+1])
    # exit()

    framerate = 60      # frames per second
    N = 256

    image = load_image()
    fs, channels = load_samples("beelze.wav")

    # fs samples per second
    # framerate frames per second
    # fs/framerate samples per frame
    # [i*fs/framerate - (N-1):i*fs/framerate+1)] samples per frame

    i=0
    while True:
        if (i-1)*int(fs/framerate)+1 > len(channels[0]):
            break
        print(i)
        #print(channels[:,i*int(fs/framerate) - (N-1):i*int(fs/framerate)+1])
        channels_magnitude = compute_FFT(channels[:,i*int(fs/framerate) - (N-1):i*int(fs/framerate)+1], N)
        # while True:
        frame = image.copy()
        frame = draw(frame, channels_magnitude, N/2, i/(2*framerate))
        i+=1
        cv2.imshow("frame", frame)
        cv2.waitKey(int(1000/framerate))






if __name__ == "__main__":
    main()