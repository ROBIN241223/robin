import os
from math import cos, sin, pi, floor
import matplotlib.pyplot as plt
from adafruit_rplidar import RPLidar
import numpy as np
import time

class LidarScanner:
    def __init__(self, port, theta_x=0, theta_y=90, neighbor_range=10, angle_resolution=0.25, pwm=800):
        self.port = port
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.neighbor_range = neighbor_range
        self.angle_resolution = angle_resolution
        self.lidar = RPLidar(None, port, timeout=6)
        self.lidar.set_pwm(pwm)
        self.lidar._max_buf_meas = 2000

        self.distance_x = 0
        self.distance_y = 0
        self.frequency = 0
        self.running = True  # Biến kiểm soát vòng lặp

    def average_distance(self, scan_data, index):
        start = max(0, index - self.neighbor_range)
        end = min(len(scan_data), index + self.neighbor_range + 1)
        valid_distances = [d for d in scan_data[start:end] if d > 0]
        return np.mean(valid_distances) if valid_distances else None  # Trả về None nếu không có dữ liệu hợp lệ

    def start_scan(self):
        print(self.lidar.info)
        self.lidar.start_motor()
        plt.ion()
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        last_time = time.time()
        scan_count = 0

        try:
            while self.running:  # Kiểm tra biến để có thể dừng vòng lặp
                ax.clear()
                ax.set_theta_zero_location('N')
                ax.set_theta_direction(-1)
                scan_data = [0] * int(360 / self.angle_resolution)

                for scan in self.lidar.iter_scans():
                    if not self.running:  # Kiểm tra nếu chương trình cần dừng
                        break

                    ax.clear()
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    scan_data = [0] * int(360 / self.angle_resolution)

                    for (_, angle, distance) in scan:
                        index = min([len(scan_data) - 1, floor(angle / self.angle_resolution)])
                        scan_data[index] = distance

                    angles = [i * self.angle_resolution for i, d in enumerate(scan_data) if d > 0]
                    distances = [d for d in scan_data if d > 0]

                    ax.plot([angle * pi / 180 for angle in angles], distances, 'bo', markersize=1)

                    index_x = int(self.theta_x / self.angle_resolution)
                    index_y = int(self.theta_y / self.angle_resolution)

                    new_distance_x = self.average_distance(scan_data, index_x)
                    new_distance_y = self.average_distance(scan_data, index_y)

                    if new_distance_x is not None:
                        self.distance_x = new_distance_x
                    if new_distance_y is not None:
                        self.distance_y = new_distance_y

                    ax.plot([self.theta_x * pi / 180, self.theta_x * pi / 180], [0, self.distance_x], color='green')
                    ax.plot([self.theta_y * pi / 180, self.theta_y * pi / 180], [0, self.distance_y], color='green')

                    fig.texts.clear()
                    fig.text(0.05, 0.15, f'Theta {self.theta_x}°: {self.distance_x:.1f} mm', fontsize=10, ha='left')
                    fig.text(0.05, 0.10, f'Theta {self.theta_y}°: {self.distance_y:.1f} mm', fontsize=10, ha='left')

                    scan_count += 1
                    current_time = time.time()
                    if current_time - last_time >= 1:
                        self.frequency = scan_count / (current_time - last_time)
                        fig.text(0.05, 0.05, f'Scan Frequency: {self.frequency:.1f} Hz', fontsize=10, ha='left')
                        last_time = current_time
                        scan_count = 0

                    plt.pause(0.00001)

        except KeyboardInterrupt:
            print('Stopping.')
        finally:
            self.stop_scan()

    def get_data(self):
        """Hàm này trả về giá trị khoảng cách và tần số quét"""
        return self.distance_x, self.distance_y, self.frequency

    def stop_scan(self):
        """Dừng quét và giải phóng tài nguyên"""
        self.running = False
        self.lidar.stop()
        self.lidar.disconnect()
        plt.ioff()
        plt.show()