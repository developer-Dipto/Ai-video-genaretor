import json
import os
import requests
import numpy as np
from PIL import Image
from moviepy.editor import ImageClip, concatenate_videoclips, vfx

# আপনার দেওয়া API কনফিগারেশন
API_URL = "https://simple-ai-image-genaretor.deptoroy91.workers.dev/"
API_KEY = "01828567716"

def generate_image(prompt, scene_n):
    print(f"Generating image for scene {scene_n}...")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    payload = {
        "prompt": prompt,
        "size": "16:9",
        "model": "@cf/black-forest-labs/flux-1-schnell"
    }
    
    response = requests.post(API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        image_path = f"scene_{scene_n}.jpg"
        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"Image saved: {image_path}")
        return image_path
    else:
        raise Exception(f"API Error for scene {scene_n}: {response.text}")

# MoviePy এর জন্য কাস্টম জুম-ইন ও জুম-আউট ফাংশন (যাতে কোয়ালিটি ভালো থাকে)
def apply_motion(clip, motion):
    w, h = clip.size
    
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        
        if motion == "zoom-in":
            # 1.0 থেকে 1.15 পর্যন্ত স্কেল আপ হবে
            scale = 1.0 + 0.15 * (t / clip.duration)
        elif motion == "zoom-out":
            # 1.15 থেকে 1.0 পর্যন্ত স্কেল ডাউন হবে
            scale = 1.15 - 0.15 * (t / clip.duration)
        else:
            return get_frame(t) # static

        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # ফ্রেমটিকে সেন্টারে ক্রপ করা
        left = (new_w - w) / 2
        top = (new_h - h) / 2
        img = img.crop((left, top, left + w, top + h))
        return np.array(img)

    if motion in ["zoom-in", "zoom-out"]:
        return clip.fl(effect)
    return clip

def create_video():
    # GitHub Action থেকে JSON ডাটা রিড করা
    with open("input.json", "r") as f:
        data = json.load(f)

    video_clips = []

    for scene in data["scenes"]:
        # ১. ইমেজ জেনারেট করা
        img_path = generate_image(scene["bg_prompt"], scene["scene_n"])
        
        # ২. ImageClip তৈরি ও Duration সেট করা
        clip = ImageClip(img_path).set_duration(scene["duration"])
        
        # ৩. Motion অ্যাপ্লাই করা
        clip = apply_motion(clip, scene["motion"])
        
        # ৪. Transition অ্যাপ্লাই করা
        if scene["transition"] == "fade-in":
            clip = clip.fadein(1.0)
        elif scene["transition"] == "fade-out":
            clip = clip.fadeout(1.0)
            
        video_clips.append(clip)

    # ৫. সবগুলো সিন একসাথে জোড়া লাগানো
    print("Concatenating clips...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    
    # ৬. ফাইনাল ভিডিও রেন্ডার করা
    print("Rendering final video...")
    final_video.write_videofile("output.mp4", fps=24, codec="libx264")
    print("Video generation complete!")

if __name__ == "__main__":
    create_video()
