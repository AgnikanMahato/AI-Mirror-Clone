import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, 
                     min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                    (25, 27), (26, 28), (27, 31), (28, 32)]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
plt.ion()

neon_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), 
               (255, 100, 255), (100, 255, 255), (255, 150, 0)]

start_time = time.time()
frame_count = 0
trail_points = []
max_trail_length = 15

def create_neon_effect(color, intensity=1.0):
    return tuple(int(c * intensity) for c in color)

def get_gradient_color(t):
    hue = (t * 60) % 360
    c = 1
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = 0
    
    if 0 <= hue < 60:
        r, g, b = c, x, 0
    elif 60 <= hue < 120:
        r, g, b = x, c, 0
    elif 120 <= hue < 180:
        r, g, b = 0, c, x
    elif 180 <= hue < 240:
        r, g, b = 0, x, c
    elif 240 <= hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))

def add_particle_effects(frame, points, color):
    overlay = np.zeros_like(frame, dtype=np.uint8)
    current_time = time.time() * 1000
    
    for i, (x, y) in enumerate(points):
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            radius = int(15 + 10 * math.sin(current_time * 0.01 + i))
            alpha = 0.3 + 0.2 * math.sin(current_time * 0.005 + i * 0.5)
            
            cv2.circle(overlay, (x, y), radius, color, -1)
            cv2.circle(overlay, (x, y), radius//2, (255, 255, 255), -1)
    
    return cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    current_time = time.time()
    wave_time = current_time - start_time
    
    gradient_color = get_gradient_color(wave_time * 0.5)
    pulse_intensity = 0.7 + 0.3 * math.sin(wave_time * 4)
    neon_color = create_neon_effect(gradient_color, pulse_intensity)
    
    if results.pose_landmarks:
        height, width, _ = frame.shape
        
        landmarks_2d = [(int(lm.x * width), int(lm.y * height)) for lm in results.pose_landmarks.landmark]
        clone_landmarks_2d = [(int((1 - lm.x) * width), int(lm.y * height)) for lm in results.pose_landmarks.landmark]
        landmarks_3d = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        
        glow_layer = np.zeros_like(frame, dtype=np.uint8)
        
        # Draw original pose
        for i, j in POSE_CONNECTIONS:
            x1, y1 = landmarks_2d[i]
            x2, y2 = landmarks_2d[j]
            
            thickness_base = int(8 + 4 * math.sin(wave_time * 3 + i))
            
            for thickness in range(thickness_base, 2, -2):
                alpha = 1.0 - (thickness_base - thickness) / thickness_base * 0.8
                line_color = tuple(int(c * alpha) for c in neon_color)
                cv2.line(glow_layer, (x1, y1), (x2, y2), line_color, thickness)
        
        # Draw mirrored clone
        for i, j in POSE_CONNECTIONS:
            x1, y1 = clone_landmarks_2d[i]
            x2, y2 = clone_landmarks_2d[j]
            
            thickness_base = int(8 + 4 * math.sin(wave_time * 3 + i))
            
            for thickness in range(thickness_base, 2, -2):
                alpha = 1.0 - (thickness_base - thickness) / thickness_base * 0.8
                clone_color = tuple(int(c * alpha * 0.7) for c in neon_color)  # Slightly dimmer clone
                cv2.line(glow_layer, (x1, y1), (x2, y2), clone_color, thickness)
        
        joint_points = []
        clone_joint_points = []
        
        # Draw original joints
        for idx, (x, y) in enumerate(landmarks_2d):
            if 0 <= x < width and 0 <= y < height:
                joint_points.append((x, y))
                
                radius = int(12 + 6 * math.sin(wave_time * 2 + idx * 0.3))
                
                for r in range(radius, 2, -2):
                    alpha = 1.0 - (radius - r) / radius * 0.7
                    circle_color = tuple(int(c * alpha) for c in neon_color)
                    cv2.circle(glow_layer, (x, y), r, circle_color, -1)
                
                cv2.circle(glow_layer, (x, y), 3, (255, 255, 255), -1)
        
        # Draw clone joints
        for idx, (x, y) in enumerate(clone_landmarks_2d):
            if 0 <= x < width and 0 <= y < height:
                clone_joint_points.append((x, y))
                
                radius = int(12 + 6 * math.sin(wave_time * 2 + idx * 0.3))
                
                for r in range(radius, 2, -2):
                    alpha = 1.0 - (radius - r) / radius * 0.7
                    circle_color = tuple(int(c * alpha * 0.7) for c in neon_color)  # Slightly dimmer clone
                    cv2.circle(glow_layer, (x, y), r, circle_color, -1)
                
                cv2.circle(glow_layer, (x, y), 3, (180, 180, 180), -1)  # Dimmer center
        
        trail_points.append(joint_points.copy() + clone_joint_points.copy())
        if len(trail_points) > max_trail_length:
            trail_points.pop(0)
        
        for t_idx, trail_frame in enumerate(trail_points[:-1]):
            trail_alpha = t_idx / len(trail_points) * 0.3
            trail_color = tuple(int(c * trail_alpha) for c in neon_color)
            
            for x, y in trail_frame:
                cv2.circle(glow_layer, (x, y), 2, trail_color, -1)
        
        frame = add_particle_effects(frame, joint_points + clone_joint_points, neon_color)
        frame = cv2.addWeighted(frame, 0.6, glow_layer, 0.8, 0)
        
        if frame_count % 3 == 0:
            ax.clear()
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_facecolor('black')
            ax.set_title("3D Cyberpunk Pose", color='cyan', fontsize=14, weight='bold')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            for i, j in POSE_CONNECTIONS:
                x_vals, y_vals, z_vals = zip(landmarks_3d[i], landmarks_3d[j])
                line_color = tuple(c/255 for c in neon_color)
                ax.plot(x_vals, y_vals, z_vals, color=line_color, linewidth=4, alpha=0.9)
            
            for x, y, z in landmarks_3d:
                ax.scatter(x, y, z, c=[line_color], s=80, alpha=0.9, edgecolors='white', linewidth=1)
            
            plt.draw()
            plt.pause(0.001)
    
    else:
        height, width, _ = frame.shape
        background = np.zeros_like(frame)
        search_text = "SCANNING FOR HUMAN..."
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(search_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        scan_color = get_gradient_color(wave_time * 2)
        cv2.putText(background, search_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, scan_color, thickness)
        
        frame = cv2.addWeighted(frame, 0.3, background, 0.7, 0)
    
    fps_text = f"FPS: {int(1/(time.time() - current_time + 0.001))}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("CYBERPUNK MOTION TRACKER", frame)
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()