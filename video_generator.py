import sys
import json
import os
import requests
import numpy as np
import random
from PIL import Image
from moviepy.editor import *
import moviepy.video.fx.all as vfx # ফাস্ট-ফরোয়ার্ড করার জন্য

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
    overlap = 2.0 # ২ সেকেন্ডের ট্রানজিশন

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
            
            if trans_file != "none":
                if os.path.exists(trans_path):
                    print(f"Applying Transition -> {trans_file} (Type: {trans_type})")
                    # FIX 1: ট্রানজিশন ভিডিওকে ফাস্ট করে ওভারল্যাপের সমান (২ সেকেন্ড) করা হলো
                    trans_video = VideoFileClip(trans_path).resize((W, H))
                    trans_video = trans_video.fx(vfx.speedx, final_duration=overlap)
                    trans_dur = overlap
                    
                    if trans_type == 1:
                        # TYPE 1: White = 1st Image, Black = 2nd Image
                        start_time = current_time - trans_dur
                        clip = clip.set_start(start_time)
                        
                        def make_mask_type1(t, get_t_frame=trans_video.get_frame, tdur=trans_dur, w=W, h=H):
                            if t < tdur:
                                try: frame = get_t_frame(t)
                                except: frame = get_t_frame(tdur - 0.01) # ফ্রেম ড্রপ ঠেকানোর সেফটি
                                gray = frame[:, :, 0] / 255.0
                                return np.clip(1.0 - gray, 0.0, 1.0)
                            return np.ones((h, w), dtype=float)
                            
                        custom_mask = VideoClip(ismask=True, make_frame=make_mask_type1).set_duration(clip.duration)
                        clip = clip.set_mask(custom_mask)
                        clips.append(clip)
                        current_time = start_time + clip.duration
                        
                    elif trans_type == 2:
                        # TYPE 2: Full screen black ink -> Then Fade in 2nd image
                        black_start = current_time - trans_dur
                        fade_in_time = 1.0
                        
                        # FIX 2: Black background কে fade_in_time পর্যন্ত লম্বা করা হলো যাতে গ্লিচ না হয়
                        black_clip = ColorClip(size=(W, H), color=(0,0,0)).set_duration(trans_dur + fade_in_time).set_start(black_start)
                        
                        def make_mask_type2(t, get_t_frame=trans_video.get_frame, tdur=trans_dur, w=W, h=H):
                            if t < tdur:
                                try: frame = get_t_frame(t)
                                except: frame = get_t_frame(tdur - 0.01)
                                gray = frame[:, :, 0] / 255.0
                                return np.clip(1.0 - gray, 0.0, 1.0)
                            return np.ones((h, w), dtype=float)
                            
                        black_mask = VideoClip(ismask=True, make_frame=make_mask_type2).set_duration(trans_dur + fade_in_time)
                        black_clip = black_clip.set_mask(black_mask)
                        clips.append(black_clip)
                        
                        clip = clip.set_start(current_time).fadein(fade_in_time)
                        clips.append(clip)
                        current_time += clip.duration
                else:
                    # FIX 3: ফাইল না পেলে লাল রঙের ওয়ার্নিং দেবে যাতে আপনি বুঝতে পারেন নাম ভুল হয়েছে
                    print(f"\n❌ WARNING: File '{trans_file}' not found in assets/ folder! Check spelling! Applying normal cut instead.\n")
                    clip = clip.set_start(current_time)
                    clips.append(clip)
                    current_time += clip.duration
            else:
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
    print("Success! Bug-Free Video generation completed.")

if __name__ == "__main__":
    build_video()
