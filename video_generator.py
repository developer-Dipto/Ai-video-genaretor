import sys
import json
import os
import requests
import numpy as np
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

# Universal Motion Engine (No Black Borders)
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

# ==========================================
# ADVANCED TRANSITION ENGINES (BUG FREE)
# ==========================================

def apply_type1_transition(base_clip, trans_path, overlap, W, H):
    """ TYPE 1: White = Prev Scene, Black = New Scene """
    trans_video = VideoFileClip(trans_path).resize((W, H))
    orig_dur = trans_video.duration
    time_mult = orig_dur / overlap # Time mapping to avoid speedx glitch

    def mask_frame(t):
        if t < overlap:
            orig_t = min(t * time_mult, orig_dur - 0.01) # Safety boundary
            frame = trans_video.get_frame(orig_t)
            gray = frame[:, :, 0] / 255.0
            return np.clip(1.0 - gray, 0.0, 1.0)
        return np.ones((H, W), dtype=float)

    mask_clip = VideoClip(ismask=True, make_frame=mask_frame).set_duration(base_clip.duration)
    return base_clip.set_mask(mask_clip)

def create_type2_black_overlay(trans_path, start_time, overlap, fade_in_time, W, H):
    """ TYPE 2: Ink paints screen black -> Prevents scene repeat glitch """
    trans_video = VideoFileClip(trans_path).resize((W, H))
    orig_dur = trans_video.duration
    time_mult = orig_dur / overlap

    def mask_frame(t):
        if t < overlap:
            orig_t = min(t * time_mult, orig_dur - 0.01)
            frame = trans_video.get_frame(orig_t)
            gray = frame[:, :, 0] / 255.0
            return np.clip(1.0 - gray, 0.0, 1.0)
        return np.ones((H, W), dtype=float)

    # Black layer stays active slightly longer to cover the fade-in of the next scene
    black_clip = ColorClip(size=(W, H), color=(0,0,0)).set_duration(overlap + fade_in_time)
    mask_clip = VideoClip(ismask=True, make_frame=mask_frame).set_duration(overlap + fade_in_time)
    return black_clip.set_mask(mask_clip).set_start(start_time)


def build_video():
    try:
        with open("input.json", "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                print("Error: input.json is empty!")
                sys.exit(1)
            data = json.loads(content)
    except Exception as e:
        print(f"JSON Error: {e}")
        sys.exit(1)

    target_ratio = data.get("global_settings", {}).get("ratio", "16:9")
    W, H = DIMENSIONS.get(target_ratio, (1920, 1080))
    
    clips =[]
    current_time = 0
    overlap = 2.0
    fade_in_time = 1.0

    for idx, scene in enumerate(data.get("scenes",[])):
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
                print(f"Applying Transition -> {trans_file} (Type: {trans_type})")
                
                if trans_type == 1:
                    start_time = current_time - overlap
                    clip = clip.set_start(start_time)
                    clip = apply_type1_transition(clip, trans_path, overlap, W, H)
                    clips.append(clip)
                    current_time = start_time + clip.duration
                    
                elif trans_type == 2:
                    # 1. Background turns to black using ink
                    black_start = current_time - overlap
                    black_overlay = create_type2_black_overlay(trans_path, black_start, overlap, fade_in_time, W, H)
                    clips.append(black_overlay)
                    
                    # 2. Smoothly fade in the new scene on top of the black ink
                    clip = clip.set_start(current_time).fadein(fade_in_time)
                    clips.append(clip)
                    current_time += clip.duration
            else:
                if trans_file != "none":
                    print(f"❌ WARNING: File '{trans_file}' not found! Applying cut.")
                clip = clip.set_start(current_time)
                clips.append(clip)
                current_time += clip.duration

    print("Compositing all layers. Rendering the masterpiece...")
    final_video = CompositeVideoClip(clips, size=(W, H))
    
    final_video.write_videofile(
        "final_video.mp4", 
        fps=30, 
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=4
    )
    print("Success! Perfect Video generation completed.")

if __name__ == "__main__":
    build_video()
