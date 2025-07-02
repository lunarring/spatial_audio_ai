import gradio as gr
import os
import random
import string
import threading
import time
import numpy as np
import soundfile as sf

# Try to import visual rendering dependencies
VISUAL_RENDERING_AVAILABLE = False
try:
    import lunar_tools as lt
    import torch
    from PIL import Image, ImageDraw
    VISUAL_RENDERING_AVAILABLE = True
except ImportError as e:
    print(f"Visual rendering not available: {e}")
    print("Audio playback will work normally without visual rendering")
    lt = None
    torch = None
    Image = None
    ImageDraw = None

from spatial_audio_ai.generators.stable_audio import (
    StableAudioOpenSmall, SoundPoolGenerator, SpatialSoundPoolPlayer
)
from spatial_audio_ai.tools.tools import apply_fade_in_out, save_sound
from spatial_audio_ai.tools.spatializer import (
    SO_Playback, SO_PlaybackCircularMove, CHUNKSIZE, SAMPLING_RATE
)

# Default prompts (hybrid/mixed)
default_prompts = [
    'forest creatures playing metallic instruments',
    'digital rainstorm on cybernetic landscape',
    'organic heartbeat with electronic processing',
    'bird calls transformed through vocoder',
    'underwater cave with resonant frequencies',
    'human breath controlling synthesizer parameters',
    'insect swarm with granular synthesis',
    'stone temple with electromagnetic resonance',
    'crackling fire with ambient string quartet',
    'shamanic ritual with analog oscillators'
]


# Global variables for visual rendering
visual_renderer = None
visual_renderer_stop_flag = False
visual_renderer_thread = None


class SpatialVisualRenderer:
    """Real-time visual renderer for spatial audio system"""
    
    def __init__(self, width=1920, height=1080, max_box_size=30):
        self.width = width
        self.height = height
        self.max_box_size = max_box_size
        self.renderer = lt.Renderer(width=width, height=height)
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Scale factor to map spatial coordinates to screen coordinates
        self.scale = min(width, height) // 3 / max_box_size
        
        # Speaker positions (same as in Spatializer class)
        self.speaker_positions = np.asarray([
            (-4.80, 4.7), (-3.0, 4.8), (-0.0, 4.8), (3.0, 4.8),
            (4.8, 4.7), (4.8, 0.4), (4.8, -4.6), (2.7, -4.6),
            (-0.0, -4.6), (-2.7, -4.6), (-4.8, -4.6), (-4.8, 0.4),
        ])
        
        # Color palette for different sound objects
        self.colors = [
            (255, 100, 100, 255),  # Red
            (100, 255, 100, 255),  # Green  
            (100, 100, 255, 255),  # Blue
            (255, 255, 100, 255),  # Yellow
            (255, 100, 255, 255),  # Magenta
            (100, 255, 255, 255),  # Cyan
            (255, 150, 100, 255),  # Orange
            (150, 100, 255, 255),  # Purple
            (100, 255, 150, 255),  # Light green
            (255, 100, 150, 255),  # Pink
        ]
        
        # Store object trails for visual history
        self.object_trails = {}
        self.max_trail_length = 50
        
    def world_to_screen(self, pos):
        """Convert world coordinates to screen coordinates"""
        screen_x = self.center_x + pos[0] * self.scale
        screen_y = self.center_y - pos[1] * self.scale  # Flip Y axis
        return int(screen_x), int(screen_y)
        
    def render_frame(self, scene, box_size):
        """Render a single frame showing all sound source positions"""
        # Create RGBA image
        img = Image.new('RGBA', (self.width, self.height), (20, 20, 30, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw spatial area boundary
        box_corners = [
            self.world_to_screen([-box_size, -box_size]),
            self.world_to_screen([box_size, -box_size]),
            self.world_to_screen([box_size, box_size]),
            self.world_to_screen([-box_size, box_size])
        ]
        draw.polygon(box_corners, outline=(80, 80, 80, 255), fill=None, width=2)
        
        # Draw speaker positions
        for i, speaker_pos in enumerate(self.speaker_positions):
            x, y = self.world_to_screen(speaker_pos)
            draw.ellipse([x-8, y-8, x+8, y+8], fill=(200, 200, 200, 255))
            # Speaker number
            try:
                draw.text((x-5, y-5), str(i+1), fill=(0, 0, 0, 255))
            except:
                pass
        
        # Draw center point
        center_screen = self.world_to_screen([0, 0])
        draw.ellipse([center_screen[0]-3, center_screen[1]-3, 
                     center_screen[0]+3, center_screen[1]+3], 
                    fill=(255, 255, 255, 255))
        
        # Draw sound objects and their trails
        active_objects = scene.sound_objects if hasattr(scene, 'sound_objects') else []
        
        for i, sound_obj in enumerate(active_objects):
            if hasattr(sound_obj, 'position'):
                # Update position for moving objects
                if hasattr(sound_obj, 'update_position'):
                    sound_obj.update_position()
                
                pos = sound_obj.position
                screen_pos = self.world_to_screen(pos)
                color = self.colors[i % len(self.colors)]
                
                # Store trail position
                obj_id = id(sound_obj)
                if obj_id not in self.object_trails:
                    self.object_trails[obj_id] = []
                self.object_trails[obj_id].append(screen_pos)
                if len(self.object_trails[obj_id]) > self.max_trail_length:
                    self.object_trails[obj_id].pop(0)
                
                # Draw trail
                if len(self.object_trails[obj_id]) > 1:
                    for j in range(1, len(self.object_trails[obj_id])):
                        alpha = int(255 * j / len(self.object_trails[obj_id]))
                        trail_color = (*color[:3], alpha)
                        prev_pos = self.object_trails[obj_id][j-1]
                        curr_pos = self.object_trails[obj_id][j]
                        draw.line([prev_pos, curr_pos], fill=trail_color, width=2)
                
                # Draw sound object
                radius = 15 if isinstance(sound_obj, SO_PlaybackCircularMove) else 10
                draw.ellipse([screen_pos[0]-radius, screen_pos[1]-radius,
                             screen_pos[0]+radius, screen_pos[1]+radius],
                            fill=color, outline=(255, 255, 255, 255), width=2)
                
                # Draw direction indicator for circular objects
                if isinstance(sound_obj, SO_PlaybackCircularMove):
                    # Draw arrow showing movement direction
                    if hasattr(sound_obj, 'direction') and hasattr(sound_obj, 'speed'):
                        elapsed_time = time.time() - sound_obj.start_time
                        angle = sound_obj.initial_angle + sound_obj.direction * sound_obj.speed * elapsed_time
                        arrow_end = (
                            screen_pos[0] + int(25 * np.cos(angle + sound_obj.direction * 0.3)),
                            screen_pos[1] - int(25 * np.sin(angle + sound_obj.direction * 0.3))
                        )
                        draw.line([screen_pos, arrow_end], fill=(255, 255, 255, 255), width=3)
        
        # Clean up old trails
        current_obj_ids = {id(obj) for obj in active_objects}
        self.object_trails = {k: v for k, v in self.object_trails.items() if k in current_obj_ids}
        
        # Add info text
        try:
            info_text = f"Active Sounds: {len(active_objects)} | Spatial Area: {box_size:.1f}x{box_size:.1f}"
            draw.text((10, 10), info_text, fill=(255, 255, 255, 255))
            
            # Legend
            legend_y = 40
            draw.text((10, legend_y), "Legend:", fill=(255, 255, 255, 255))
            draw.text((10, legend_y + 20), "○ Static Sound", fill=(255, 255, 255, 255))
            draw.text((10, legend_y + 40), "◉ Moving Sound", fill=(255, 255, 255, 255))
            draw.text((10, legend_y + 60), "□ Speakers", fill=(200, 200, 200, 255))
            draw.text((10, legend_y + 80), "▢ Spatial Area", fill=(80, 80, 80, 255))
        except:
            pass
        
        # Convert PIL image to torch tensor
        img_array = np.array(img).astype(np.float32)
        img_tensor = torch.from_numpy(img_array)
        
        return img_tensor

def start_visual_renderer(scene, box_size, fps=30):
    """Start the visual renderer in a separate thread"""
    global visual_renderer, visual_renderer_stop_flag, visual_renderer_thread
    
    if not VISUAL_RENDERING_AVAILABLE:
        print("Visual rendering not available - skipping")
        return
    
    visual_renderer_stop_flag = False
    visual_renderer = SpatialVisualRenderer()
    
    def render_loop():
        frame_time = 1.0 / fps
        while not visual_renderer_stop_flag:
            try:
                frame = visual_renderer.render_frame(scene, box_size)
                visual_renderer.renderer.render(frame)
                time.sleep(frame_time)
            except Exception as e:
                print(f"Rendering error: {e}")
                time.sleep(0.1)
    
    visual_renderer_thread = threading.Thread(target=render_loop)
    visual_renderer_thread.daemon = True
    visual_renderer_thread.start()
    
def stop_visual_renderer():
    """Stop the visual renderer"""
    global visual_renderer_stop_flag, visual_renderer_thread
    visual_renderer_stop_flag = True
    if visual_renderer_thread and visual_renderer_thread.is_alive():
        visual_renderer_thread.join(timeout=2)

def random_dir(base="/tmp/soundpool_gradio_"):
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    dir_tmp = f"{base}{rand}"
    os.makedirs(dir_tmp, exist_ok=True)
    return dir_tmp

# Threaded sound generation with progress callback
def generate_sounds(prompts, n_sounds, min_duration, max_duration,
                   progress=gr.Progress(track_tqdm=True)):
    dir_tmp = random_dir()
    audio_diffusion = StableAudioOpenSmall(steps=8, cfg_scale=1.0, force_mono=True)
    spg = SoundPoolGenerator(audio_diffusion, directory=dir_tmp)
    spg.set_min_duration_sound(min_duration)
    spg.set_max_duration_sound(max_duration)
    prompts = [p.strip() for p in prompts.split('\n') if p.strip()]
    n_sounds = int(n_sounds)
    for i in progress.tqdm(range(n_sounds), desc="Generating sounds"):
        prompt = random.choice(prompts)
        duration = random.uniform(
            spg.min_duration_sound, spg.max_duration_sound
        )
        seed = audio_diffusion.set_random_seed()
        audio_diffusion.set_seed(seed)
        audio_diffusion.set_audio_end_in_s(duration)
        sound = audio_diffusion.generate_sound(prompt)
        sound = apply_fade_in_out(sound, audio_diffusion.sampling_rate)
        filename = f"{prompt.replace(' ', '_')[:40]}_{seed}.wav"
        file_path = os.path.join(dir_tmp, filename)
        save_sound(sound, file_path, audio_diffusion.sampling_rate)
    return dir_tmp

# Global for playback thread and stop flag
playback_thread = None
playback_stop_flag = False

def play_spatial_soundpool(
    dir_path, p_inject, box_size, volume, duration_minutes,
    use_circular, circular_radius_min, circular_radius_max,
    circular_speed_min, circular_speed_max,
    circular_angle_min, circular_angle_max,
    circular_direction, circular_center_x, circular_center_y,
    circular_loop, enable_visual_rendering=True,
    progress=gr.Progress(track_tqdm=True)
):
    global playback_thread, playback_stop_flag
    playback_stop_flag = False
    name_space = os.path.basename(dir_path)
    base_dir = os.path.dirname(dir_path)
    player = SpatialSoundPoolPlayer(
        name_space=name_space,
        base_dir=base_dir,
        p_inject=p_inject,
        box_size=box_size,
        volume=volume,
        use_circular=use_circular,
        circular_radius_range=(circular_radius_min, circular_radius_max),
        circular_speed_range=(circular_speed_min, circular_speed_max),
        circular_angle_range=(circular_angle_min, circular_angle_max),
        circular_direction_choices=[circular_direction],
        circular_center=np.array([circular_center_x, circular_center_y], dtype=float),
        circular_loop=circular_loop
    )

    # Initialize visual renderer in main thread if enabled
    visual_renderer = None
    if enable_visual_rendering and VISUAL_RENDERING_AVAILABLE:
        try:
            visual_renderer = SpatialVisualRenderer()
            print("Visual renderer initialized in main thread")
        except Exception as ve:
            print(f"Could not start visual renderer: {ve}")
            visual_renderer = None

    def playback():
        nonlocal visual_renderer  # Access outer scope variable
        last_visual_update = 0
        visual_frame_interval = 1.0 / 30  # 30 FPS
        
        try:
            player.start_initial_sound()
            
            start_time = time.time()
            duration_seconds = duration_minutes * 60
            
            # Run audio + visual in main thread (no separate threading)
            for j, chunk in enumerate(player.scene.run()):
                if playback_stop_flag:
                    print("Playback stopped by user (flag).")
                    break
                    
                # Handle audio injection
                if np.random.rand() < player.p_inject:
                    player.wav_files = [
                        f for f in os.listdir(player.dir_scan)
                        if f.endswith('.wav')
                    ]
                    random_file = random.choice(player.wav_files)
                    sound = sf.read(f"{player.dir_scan}{random_file}")[0]
                    position = np.random.uniform(
                        -player.box_size, player.box_size, size=2
                    )
                    player.scene.register(SO_Playback(sound, position=position))
                    print(f"Injected: {random_file} at position: "
                          f"({position[0]:.1f}, {position[1]:.1f})")
                
                # Send audio
                chunk = np.clip(chunk, -1, 1)
                player.sound_streamer.send(chunk)
                
                # Update visual rendering at reduced frame rate
                current_time = time.time()
                if (visual_renderer and 
                    current_time - last_visual_update > visual_frame_interval):
                    try:
                        frame = visual_renderer.render_frame(player.scene, box_size)
                        visual_renderer.renderer.render(frame)
                        last_visual_update = current_time
                    except Exception as ve:
                        print(f"Visual rendering error: {ve}")
                        visual_renderer = None  # Disable on error
                
                time.sleep(CHUNKSIZE/SAMPLING_RATE - 0.01)
                elapsed = time.time() - start_time
                if elapsed > duration_seconds:
                    print(f"Playback completed after {elapsed:.1f} seconds")
                    break
                if j % 430 == 0:
                    active_sounds = len(player.scene.sound_objects)
                    status = f"Time: {elapsed:.1f}s | Active sounds: {active_sounds}"
                    if visual_renderer:
                        status += " | Visual: ON"
                    print(status)
        except Exception as e:
            print(f"Playback error: {e}")

    # Run playback directly in main thread to avoid SDL2 threading issues
    try:
        playback()
        return f"Playback completed for {duration_minutes} minutes."
    except Exception as e:
        return f"Playback error: {e}"

def stop_playback():
    global playback_stop_flag
    playback_stop_flag = True
    return "Playback stop requested."


# Visual rendering is now integrated with audio playback

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Spatial Sound Pool Generator & Player
    Generate a pool of AI sounds from prompts, then spatially play them back!
    """
    )
    
    # Create shared state for directory
    dir_state = gr.State("")
    
    with gr.Tab("1. Generate Sound Pool"):
        prompts_box = gr.Textbox(
            label="Prompts (one per line)",
            value="\n".join(default_prompts),
            lines=10
        )
        n_sounds = gr.Number(
            label="Number of sounds to generate", value=10, precision=0
        )
        min_duration = gr.Slider(2, 20, value=3, step=0.1,
                                 label="Minimum Sound Duration (seconds)")
        max_duration = gr.Slider(2, 20, value=8, step=0.1,
                                 label="Maximum Sound Duration (seconds)")
        generate_btn = gr.Button("Generate Sounds")
        output_dir = gr.Textbox(label="Output Directory", interactive=False)
        gen_progress = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("2. Spatial Playback + Visual"):
        gr.Markdown(
            """
        Select a generated sound pool directory and play it back with spatialization.
        **NEW**: Visual rendering window will automatically open showing real-time 
        sound source positions, movement trails, and speaker locations in 2D space!
        """
        )
        dir_input = gr.Textbox(
            label="Sound Pool Directory",
            placeholder="Paste output directory from above"
        )
        p_inject = gr.Slider(0, 1, value=0.3, label="Injection Probability")
        box_size = gr.Slider(5, 30, value=15, label="Spatial Area Size")
        volume = gr.Slider(0, 1, value=0.2, label="Volume")
        duration_minutes = gr.Number(
            label="Playback Duration (minutes)", value=2, precision=0
        )
        use_circular = gr.Checkbox(
            label="Use Circular Moving Sound Objects", value=False
        )
        circular_radius_min = gr.Slider(
            0.5, 30, value=2, step=0.1, label="Circular Radius Min"
        )
        circular_radius_max = gr.Slider(
            0.5, 30, value=10, step=0.1, label="Circular Radius Max"
        )
        circular_speed_min = gr.Slider(
            0.01, 5, value=0.2, step=0.01, label="Circular Speed Min (rad/s)"
        )
        circular_speed_max = gr.Slider(
            0.01, 5, value=1.0, step=0.01, label="Circular Speed Max (rad/s)"
        )
        circular_angle_min = gr.Slider(
            0, 6.283, value=0, step=0.01, label="Circular Angle Min (rad)"
        )
        circular_angle_max = gr.Slider(
            0, 6.283, value=6.283, step=0.01, label="Circular Angle Max (rad)"
        )
        circular_direction = gr.Dropdown(
            [-1, 1], value=1, label="Circular Direction (1=CCW, -1=CW)"
        )
        circular_center_x = gr.Slider(
            -30, 30, value=0, step=0.1, label="Circular Center X"
        )
        circular_center_y = gr.Slider(
            -30, 30, value=0, step=0.1, label="Circular Center Y"
        )
        circular_loop = gr.Checkbox(
            label="Loop Circular Sound", value=True
        )
        enable_visual = gr.Checkbox(
            label="Enable Visual Rendering", 
            value=VISUAL_RENDERING_AVAILABLE,
            interactive=VISUAL_RENDERING_AVAILABLE
        )
        
        with gr.Row():
            play_btn = gr.Button("Start Spatial Playback + Visual")
            stop_btn = gr.Button("Stop Playback")
        
        playback_status = gr.Textbox(label="Playback Status", interactive=False)
        visual_status = gr.Textbox(label="Visual Status", interactive=False,
                                   value="Visual rendering available" if VISUAL_RENDERING_AVAILABLE 
                                   else "Visual rendering not available (install SDL2)")

        play_btn.click(
            play_spatial_soundpool,
            inputs=[
                dir_input, p_inject, box_size, volume, duration_minutes,
                use_circular, circular_radius_min, circular_radius_max,
                circular_speed_min, circular_speed_max,
                circular_angle_min, circular_angle_max,
                circular_direction, circular_center_x, circular_center_y,
                circular_loop, enable_visual
            ],
            outputs=playback_status
        )
        stop_btn.click(stop_playback, outputs=playback_status)

    # Define the generate function and connect it to the button
    def _generate(prompts, n_sounds, min_duration, max_duration,
                  progress=gr.Progress(track_tqdm=True)):
        dir_tmp = generate_sounds(prompts, n_sounds, min_duration,
                                  max_duration, progress)
        return dir_tmp, f"Generated {n_sounds} sounds in {dir_tmp}", dir_tmp

    generate_btn.click(
        _generate, inputs=[prompts_box, n_sounds, min_duration, max_duration],
        outputs=[output_dir, gen_progress, dir_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="10.40.49.109")