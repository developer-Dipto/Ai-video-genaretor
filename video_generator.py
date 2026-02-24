import sys
import json
import os
import requests
import numpy as np
import random
from PIL import Image
from moviepy.editor import *

# API Config
API_URL = "https://simple-ai-image-genaretor.deptoroy91.workers.dev/"
API_KEY = "01828567716"

DIMENSIONS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920)
}

def generate_image(prompt, size_ratio, scene_n):
    print(f"Generating Image for Scene {scene_n}...")
    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}
    payload = {"prompt": prompt, "size": size_ratio, "model": "@cf/black-forest-labs/flux-1-schnell"}
    response = requests.post(API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        img_path = f"scene_{scene_n}.jpg"
        with open(img_path, "wb") as f:
            f.write(response.content)
        return img_path
    else:
        raise Exception(f"API Error: {response.text}")

# Universal Motion Engine
def apply_motion(clip, motion_type):
    w, h = clip.size
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        progress = t / clip.duration
        scale = 1.15 
        
        if motion_type == "zoom-in": scale = 1.0 + 0.15 * progress
        elif motion_type == "zoom-out": scale = 1.15 - 0.15 * progress

        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        center_x, center_y = (new_w - w) / 2, (new_h - h) / 2
        pan_x, pan_y = center_x, center_y

        if motion_type == "pan-right": pan_x = (new_w - w) * progress
        elif motion_type == "pan-left": pan_x = (new_w - w) * (1 - progress)
        elif motion_type == "pan-down": pan_y = (new_h - h) * progress
        elif motion_type == "pan-up": pan_y = (new_h - h) * (1 - progress)

        img = img.crop((pan_x, pan_y, pan_x + w, pan_y + h))
        return np.array(img)
    return clip.fl(effect)

def build_video():
    # 1. Error Handling for JSON Input
    try:
        with open("input.json", "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                print("Error: input.json is empty! Please check GitHub Actions input.")
                sys.exit(1)
            data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON Format! Details: {e}")
        print(f"File content was:\n{content}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error while reading JSON: {e}")
        sys.exit(1)

    target_ratio = data.get("global_settings", {}).get("ratio", "16:9")
    W, H = DIMENSIONS.get(target_ratio, (1920, 1080))
    
    clips = []
    current_time = 0
    overlap = 2.0 # Transition duration 2 seconds

    # 2. Process Scenes
    for idx, scene in enumerate(data.get("scenes", [])):
        img_path = generate_image(scene["bg_prompt"], target_ratio, scene["scene_n"])
        base_clip = ImageClip(img_path).set_duration(scene["duration"]).resize((W, H))
        clip = apply_motion(base_clip, scene.get("motion", "static"))
        
        if idx == 0:
            clip = clip.set_start(current_time)
            clips.append(clip)
            current_time += clip.duration
        else:
            prev_scene = data["scenes"][idx-1]
            trans_file = prev_scene.get("transition_file", "none")
            trans_type = prev_scene.get("transition_type", 1)
            trans_path = f"assets/{trans_file}"
            
            if trans_file != "none" and os.path.exists(trans_path):
                trans_video = VideoFileClip(trans_path).resize((W, H))
                trans_dur = min(overlap, trans_video.duration)
                
                if trans_type == 1:
                    # TYPE 1 Logic: 2nd Image appears in the Black Ink area of transition
                    start_time = current_time - trans_dur
                    clip = clip.set_start(start_time)
                    
                    def make_mask_type1(t, get_t_frame=trans_video.get_frame, tdur=trans_dur, w=W, h=H):
                        if t < tdur:
                            frame = get_t_frame(t)
                            gray = frame[:, :, 0] / 255.0
                            return 1.0 - gray # Invert: Shows 2nd clip where transition is Black
                        return np.ones((h, w))
                        
                    custom_mask = VideoClip(ismask=True, make_frame=make_mask_type1).set_duration(clip.duration)
                    clip = clip.set_mask(custom_mask)
                    clips.append(clip)
                    current_time = start_time + clip.duration
                    
                elif trans_type == 2:
                    # TYPE 2 Logic: Screen covered in Black Ink -> Fade in 2nd image
                    black_start = current_time - trans_dur
                    black_clip = ColorClip(size=(W, H), color=(0,0,0)).set_duration(trans_dur).set_start(black_start)
                    
                    def make_mask_type2(t, get_t_frame=trans_video.get_frame, tdur=trans_dur, w=W, h=H):
                        if t < tdur:
                            frame = get_t_frame(t)
                            gray = frame[:, :, 0] / 255.0
                            return 1.0 - gray # Invert to apply black color over ink
                        return np.ones((h, w))
                        
                    black_mask = VideoClip(ismask=True, make_frame=make_mask_type2).set_duration(trans_dur)
                    black_clip = black_clip.set_mask(black_mask)
                    clips.append(black_clip)
                    
                    clip = clip.set_start(current_time).fadein(1.0)
                    clips.append(clip)
                    current_time += clip.duration
            else:
                # No valid transition file found -> Simple Cut
                clip = clip.set_start(current_time)
                clips.append(clip)
                current_time += clip.duration

    print("Compositing all layers. Rendering the masterpiece...")
    final_video = CompositeVideoClip(clips, size=(W, H))
    
    # 3. Final Output
    final_video.write_videofile(
        "final_video.mp4", 
        fps=30, 
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=4
    )
    print("Success! Video generation completed.")

if __name__ == "__main__":
    build_video()
