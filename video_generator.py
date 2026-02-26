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

# Universal Motion Engine (Continuous Smooth Motion)
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
    
    video_clips =[]
    current_time = 0.0
    overlap = 2.0  # 2 Seconds Transition Time

    scenes = data.get("scenes",[])
    
    for idx, scene in enumerate(scenes):
        is_last_scene = (idx == len(scenes) - 1)
        base_duration = float(scene.get("duration", 5.0))
        
        # আপনার লজিক অনুযায়ী: Clip 1-এর টোটাল ডিউরেশন হবে base_duration + overlap/2
        # যাতে ট্রানজিশন চলাকালীন ক্লিপটি ব্যাকগ্রাউন্ডে উপস্থিত থাকে
        if not is_last_scene:
            total_clip_duration = base_duration + (overlap / 2)
        else:
            total_clip_duration = base_duration
            
        # ইমেজ জেনারেট এবং মোশন অ্যাপ্লাই (মোশন পুরো সময় জুড়ে চলবে, তাই কোনো অ্যানিমেশন জাম্প হবে না)
        img_path = generate_image(scene["bg_prompt"], target_ratio, scene["scene_n"])
        base_clip = ImageClip(img_path).set_duration(total_clip_duration).resize((W, H))
        clip = apply_motion(base_clip, scene.get("motion", "static"))
        
        # ক্লিপটি টাইমলাইনের current_time এ বসানো হলো
        clip = clip.set_start(current_time)

        if not is_last_scene:
            trans_file = scene.get("transition_file", "none")
            trans_path = f"assets/{trans_file}"
            
            if trans_file != "none" and os.path.exists(trans_path):
                print(f"Applying Luma Matte Transition -> {trans_file}")
                
                # 1. Load MP4 mathematically to avoid speedx FFMPEG bugs
                trans_video = VideoFileClip(trans_path).resize((W, H))
                trans_orig_dur = trans_video.duration
                time_mult = trans_orig_dur / overlap  # 5s -> 2s (Multiplier = 2.5)
                
                def make_luma_mask(t, t_vid=trans_video, t_mult=time_mult, t_dur=trans_orig_dur, c_dur=total_clip_duration, over=overlap, w=W, h=H):
                    # ট্রানজিশন শুরু হওয়ার আগে মাস্ক পুরোপুরি সাদা (1.0) থাকবে, ফলে 1st Image পুরোপুরি দেখা যাবে
                    if t < (c_dur - over):
                        return np.ones((h, w), dtype=float)
                    
                    # ট্রানজিশন শুরু হলে (শেষের ২ সেকেন্ডে)
                    elif t <= c_dur:
                        trans_t = (t - (c_dur - over)) * t_mult
                        trans_t = min(trans_t, t_dur - 0.01) # Safety bound
                        try:
                            frame = t_vid.get_frame(trans_t)
                        except:
                            frame = t_vid.get_frame(t_dur - 0.01)
                            
                        # সাদা = 1 (1st Image), কালো = 0 (2nd Image), গ্রে = Smooth Blend
                        gray = frame[:, :, 0] / 255.0
                        return np.clip(gray, 0.0, 1.0)
                        
                    return np.zeros((h, w), dtype=float)

                mask_clip = VideoClip(ismask=True, make_frame=make_luma_mask).set_duration(total_clip_duration)
                clip = clip.set_mask(mask_clip)
                
            else:
                if trans_file != "none":
                    print(f"❌ WARNING: File '{trans_file}' not found! Applying hard cut.")
                    
            # 2nd Image এর Start Time (1st Image এর base duration এর মাঝে overlap/2 বিয়োগ করে)
            current_time += base_duration - (overlap / 2)
            
        video_clips.append(clip)

    print("Compositing all layers. Reversing Z-Index for Professional Luma Matte...")
    
    # সবচেয়ে গুরুত্বপূর্ণ ফিক্স: 
    # Clip 1 কে উপরে রাখতে হবে, Clip 2 কে নিচে। তাই লিস্টটাকে উল্টে (Reverse) দেওয়া হলো।
    # এর ফলে 1st Image এর কালো মাস্কের ভেতর দিয়ে নিচের 2nd Image ঠিকমতো দেখা যাবে।
    reversed_clips = video_clips[::-1]
    
    final_video = CompositeVideoClip(reversed_clips, size=(W, H))
    
    final_video.write_videofile(
        "final_video.mp4", 
        fps=30, 
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=4
    )
    print("Success! Perfect Professional Video generation completed without a single glitch.")

if __name__ == "__main__":
    build_video()
