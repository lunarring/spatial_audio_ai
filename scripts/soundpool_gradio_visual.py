import gradio as gr
import os
import random
import string
import time
import threading
import numpy as np
import soundfile as sf

from spatial_audio_ai.generators.stable_audio import (
    StableAudioOpenSmall, SoundPoolGenerator, SpatialSoundPoolPlayer
)
from spatial_audio_ai.tools.tools import apply_fade_in_out, save_sound
from spatial_audio_ai.tools.spatializer import (
    SO_Playback, SO_PlaybackCircularMove, CHUNKSIZE, SAMPLING_RATE
)

# Try to import pygame for visualization
VISUAL_RENDERING_AVAILABLE = False
try:
    import pygame
    VISUAL_RENDERING_AVAILABLE = True
    print("Pygame visualization available")
except ImportError as e:
    print(f"Visual rendering not available: {e}")
    print("Audio playback will work normally without visual rendering")
    pygame = None

# Default prompts (hybrid/mixed)
default_prompts = [
    "Dripping water echoing into a shallow stone pool.",
    "Slow, reverberant footsteps on damp cavern floor.",
    "Gentle underground stream flowing over pebbles.",
    "Breeze weaving through stalactites like chimes.",
    "Distant bat echolocation clicks and wing flutters.",
    "Low, resonant drone with shifting loose stones.",
    "Sharp crystalline tones as stone taps crystal.",
    "Soft, pulsing rumble of the cavern breathing.",
    "Faint human whispers carried by echoing walls.",
    "Distant waterfall roar swelling then receding."
]



class PygameVisualRenderer:
    """Clean pygame-based visual renderer for spatial audio"""
    
    def __init__(self, width=1000, height=700):
        self.width = width
        self.height = height
        self.running = False
        self.screen = None
        self.clock = None
        
        # Center and scale
        self.center_x = width // 2
        self.center_y = height // 2
        self.scale = min(width, height) // 8
        
        # Speaker positions (12-speaker surround)
        self.speaker_positions = np.array([
            (-4.80, 4.7), (-3.0, 4.8), (-0.0, 4.8), (3.0, 4.8),
            (4.8, 4.7), (4.8, 0.4), (4.8, -4.6), (2.7, -4.6),
            (-0.0, -4.6), (-2.7, -4.6), (-4.8, -4.6), (-4.8, 0.4),
        ])
        
        # Colors for sound objects
        self.colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
            (255, 180, 100),  # Orange
            (180, 100, 255),  # Purple
            (100, 255, 180),  # Light green
            (255, 100, 180),  # Pink
        ]
        
        # Trail storage
        self.trails = {}
        self.max_trail_length = 40
        
        # Font
        self.font = None
        
        # Scene data snapshot (thread-safe)
        self.scene_snapshot = []
        self.snapshot_lock = threading.Lock()
        
    def initialize(self):
        """Initialize pygame"""
        if not pygame:
            return False
            
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Spatial Audio Visualization")
        self.clock = pygame.time.Clock()
        
        try:
            self.font = pygame.font.Font(None, 24)
        except:
            self.font = None
            
        self.running = True
        return True
        
    def world_to_screen(self, world_pos):
        """Convert world coordinates to screen pixels"""
        x = int(self.center_x + world_pos[0] * self.scale)
        y = int(self.center_y - world_pos[1] * self.scale)  # Flip Y
        return (x, y)
    
    def update_frame(self, box_size):
        """Update and draw a single frame using snapshot data"""
        if not self.running or not self.screen:
            return False
            
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
        
        # Clear screen with dark background
        self.screen.fill((20, 25, 35))
        
        # Draw spatial area boundary
        self._draw_boundary(box_size)
        
        # Draw center point
        center_screen = self.world_to_screen((0, 0))
        pygame.draw.circle(self.screen, (255, 255, 255), center_screen, 4)
        
        # Get scene snapshot safely
        with self.snapshot_lock:
            current_snapshot = list(self.scene_snapshot)
        
        # Update positions and draw objects
        self._update_and_draw_objects_from_snapshot(current_snapshot)
        
        # Draw info panel
        self._draw_info_panel(current_snapshot, box_size)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS to reduce load
        
        return True
    
    def _draw_boundary(self, box_size):
        """Draw spatial area boundary"""
        corners = [
            self.world_to_screen((-box_size, -box_size)),
            self.world_to_screen((box_size, -box_size)),
            self.world_to_screen((box_size, box_size)),
            self.world_to_screen((-box_size, box_size))
        ]
        pygame.draw.polygon(self.screen, (100, 120, 150), corners, 2)
    

    def _update_and_draw_objects_from_snapshot(self, snapshot):
        """Draw objects from snapshot data - positions are already correct"""
        for i, obj_data in enumerate(snapshot):
            # Use the actual position from the audio system (no need to update)
            pos = obj_data['position']
            screen_pos = self.world_to_screen(pos)
            color = self.colors[i % len(self.colors)]
            
            # Update trail
            obj_id = obj_data['id']
            if obj_id not in self.trails:
                self.trails[obj_id] = []
            self.trails[obj_id].append(screen_pos)
            if len(self.trails[obj_id]) > self.max_trail_length:
                self.trails[obj_id].pop(0)
            
            # Draw trail
            self._draw_trail(obj_id, color)
            
            # Draw sound object
            radius = 15 if obj_data['is_moving'] else 10
            
            # Draw with glow effect
            for r in range(radius + 5, radius - 1, -1):
                alpha = max(50, 255 - (radius + 5 - r) * 40)
                glow_color = tuple(min(255, c + alpha // 5) for c in color)
                pygame.draw.circle(self.screen, glow_color, screen_pos, r)
            
            # Main circle
            pygame.draw.circle(self.screen, color, screen_pos, radius)
            pygame.draw.circle(self.screen, (255, 255, 255), 
                             screen_pos, radius, 2)
        
        # Clean up old trails
        active_ids = {obj_data['id'] for obj_data in snapshot}
        self.trails = {k: v for k, v in self.trails.items() 
                       if k in active_ids}
    
    def _draw_trail(self, obj_id, color):
        """Draw movement trail"""
        trail = self.trails[obj_id]
        if len(trail) < 2:
            return
        
        for i in range(1, len(trail)):
            # Fade trail based on age
            alpha = i / len(trail)
            trail_color = tuple(int(c * alpha * 0.8) for c in color)
            width = max(1, int(4 * alpha))
            
            if i < len(trail):
                pygame.draw.line(self.screen, trail_color, 
                               trail[i-1], trail[i], width)
    
    def _draw_info_panel(self, snapshot, box_size):
        """Draw information panel"""
        if not self.font:
            return
            
        info_lines = [
            f"Active Sounds: {len(snapshot)}",
            f"Spatial Area: {box_size:.1f}x{box_size:.1f}",
            "",
            "Sound Objects:",
        ]
        
        # Add info about each object
        for i, obj_data in enumerate(snapshot[:6]):  # Show first 6 objects
            obj_type = "●" if obj_data['is_moving'] else "○"
            pos = obj_data['position']
            info_lines.append(f"{obj_type} #{obj_data['index']}: "
                            f"({pos[0]:.1f},{pos[1]:.1f})")
        
        if len(snapshot) > 6:
            info_lines.append(f"... and {len(snapshot)-6} more")
        
        y_pos = 20
        for line in info_lines:
            if line:  # Skip empty lines
                text = self.font.render(line, True, (200, 200, 200))
                self.screen.blit(text, (20, y_pos))
            y_pos += 22
    
    def cleanup(self):
        """Clean up pygame resources"""
        self.running = False
        if pygame:
            pygame.quit()

    def update_scene_data(self, sound_objects):
        """Thread-safe update of scene data"""
        with self.snapshot_lock:
            self.scene_snapshot = []
            print(f"DEBUG: Scene has {len(sound_objects)} total sound objects")
            
            for i, obj in enumerate(sound_objects):
                print(f"  Object {i}: {type(obj).__name__}")
                
                # IMPORTANT: Call update_position() on ALL objects since all are dynamic
                if hasattr(obj, 'update_position'):
                    obj.update_position()  # Update position for ALL objects
                    print(f"    Dynamic object - position updated to: {obj.position}")
                
                # Now get the current position from the sound object
                if hasattr(obj, 'position') and obj.position is not None:
                    # Create a safe copy of object data with ACTUAL position
                    obj_data = {
                        'position': np.copy(obj.position),
                        'is_moving': isinstance(obj, SO_PlaybackCircularMove),
                        'id': id(obj),
                        'obj_type': type(obj).__name__,
                        'index': i  # Add index for debugging
                    }
                    
                    # For moving objects, copy their parameters for trail rendering
                    if obj_data['is_moving']:
                        obj_data.update({
                            'radius': getattr(obj, 'radius', 5.0),
                            'speed': getattr(obj, 'speed', 1.0),
                            'direction': getattr(obj, 'direction', 1),
                            'center': getattr(obj, 'center', 
                                            np.array([0.0, 0.0]))
                        })
                    
                    self.scene_snapshot.append(obj_data)
                    print(f"    ✓ Added to visualization: pos={obj_data['position']}")
                else:
                    print("    ✗ No position attribute - skipping")
            
            print(f"DEBUG: Visualization will show {len(self.scene_snapshot)} objects")


# Global visualization control
visual_renderer = None
visual_thread = None
visual_stop_flag = False


def start_pygame_visualization(box_size):
    """Start pygame visualization in separate thread"""
    global visual_renderer, visual_thread, visual_stop_flag
    
    if not VISUAL_RENDERING_AVAILABLE:
        print("Pygame not available for visualization")
        return False
    
    visual_stop_flag = False
    visual_renderer = PygameVisualRenderer()
    
    def visual_loop():
        if not visual_renderer.initialize():
            print("Failed to initialize pygame visualization")
            return
            
        print("Pygame visualization started")
        
        while not visual_stop_flag and visual_renderer.running:
            try:
                if not visual_renderer.update_frame(box_size):
                    break
                # Reduced frequency to not interfere with audio
                time.sleep(0.1)  # 10 FPS
            except Exception as e:
                print(f"Visualization error: {e}")
                # Don't break on errors, just continue
                time.sleep(0.1)
        
        visual_renderer.cleanup()
        print("Pygame visualization stopped")
    
    visual_thread = threading.Thread(target=visual_loop)
    visual_thread.daemon = True
    visual_thread.start()
    return True


def stop_pygame_visualization():
    """Stop pygame visualization"""
    global visual_stop_flag, visual_renderer
    visual_stop_flag = True
    if visual_renderer:
        visual_renderer.running = False


def random_dir(base="/tmp/soundpool_gradio_"):
    """Generate random temporary directory"""
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    dir_tmp = f"{base}{rand}"
    os.makedirs(dir_tmp, exist_ok=True)
    return dir_tmp


def generate_sounds(prompts, n_sounds, min_duration, max_duration,
                   progress=gr.Progress(track_tqdm=True)):
    """Generate sound pool with progress tracking"""
    dir_tmp = random_dir()
    audio_diffusion = StableAudioOpenSmall(steps=8, cfg_scale=1.0, 
                                          force_mono=True)
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


# Global playback control
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
    """Main playback function with pygame visualization"""
    global playback_stop_flag
    playback_stop_flag = False
    
    # Stop any existing visualization
    stop_pygame_visualization()
    
    # Setup audio player
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
        circular_center=np.array([circular_center_x, circular_center_y], 
                                dtype=float),
        circular_loop=circular_loop
    )

    # Start visualization if enabled
    visual_started = False
    if enable_visual_rendering and VISUAL_RENDERING_AVAILABLE:
        visual_started = start_pygame_visualization(box_size)
        if visual_started:
            print("Pygame visualization window opened")
        else:
            print("Failed to start visualization")

    # Main playback loop
    try:
        player.start_initial_sound()
        start_time = time.time()
        duration_seconds = duration_minutes * 60
        last_visual_update = 0
        visual_update_interval = 0.2  # Update visualization every 200ms
        
        for j, chunk in enumerate(player.scene.run()):
            if playback_stop_flag:
                print("Playback stopped by user")
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
            
            # Send audio chunk
            chunk = np.clip(chunk, -1, 1)
            player.sound_streamer.send(chunk)
            
            # Update visualization data periodically (thread-safe)
            current_time = time.time()
            if (visual_started and 
                current_time - last_visual_update > visual_update_interval):
                try:
                    # Update visualization with current scene data
                    sound_objects = (player.scene.sound_objects 
                                   if hasattr(player.scene, 'sound_objects') 
                                   else [])
                    update_visualization_data(sound_objects)
                    last_visual_update = current_time
                except Exception as ve:
                    print(f"Visualization data update error: {ve}")
            
            # Timing
            time.sleep(CHUNKSIZE/SAMPLING_RATE - 0.01)
            elapsed = time.time() - start_time
            if elapsed > duration_seconds:
                print(f"Playback completed after {elapsed:.1f} seconds")
                break
            
            # Status updates
            if j % 430 == 0:
                active_sounds = len(player.scene.sound_objects)
                status = f"Time: {elapsed:.1f}s | Active: {active_sounds}"
                if visual_started:
                    status += " | Visual: ON"
                print(status)
        
        return f"Playback completed for {duration_minutes} minutes."
    
    except Exception as e:
        print(f"Playback error: {e}")
        return f"Playback error: {e}"
    
    finally:
        # Stop visualization when playback ends
        stop_pygame_visualization()


def stop_playback():
    """Stop the current playback and visualization"""
    global playback_stop_flag
    playback_stop_flag = True
    stop_pygame_visualization()
    return "Playback and visualization stopped."


def update_visualization_data(sound_objects):
    """Update visualization with current scene data (thread-safe)"""
    global visual_renderer
    if visual_renderer:
        visual_renderer.update_scene_data(sound_objects)


# Gradio interface
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
        Select a generated sound pool directory and play it back with 
        spatialization. **NEW PYGAME VISUALIZATION**: A separate window 
        will open showing real-time sound source positions and movements!
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
            0, 6.283, value=6.283, step=0.01, 
            label="Circular Angle Max (rad)"
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
            label="Enable Pygame Visualization", 
            value=VISUAL_RENDERING_AVAILABLE,
            interactive=VISUAL_RENDERING_AVAILABLE
        )
        
        with gr.Row():
            play_btn = gr.Button("Start Spatial Playback + Pygame Visual")
            stop_btn = gr.Button("Stop Playback")
        
        playback_status = gr.Textbox(label="Playback Status", 
                                    interactive=False)
        visual_status = gr.Textbox(
            label="Visual Status", 
            interactive=False,
            value=("Pygame visualization available" 
                   if VISUAL_RENDERING_AVAILABLE 
                   else "Pygame not available (pip install pygame)")
        )

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

    # Connect generation to playback
    def _generate(prompts, n_sounds, min_duration, max_duration,
                  progress=gr.Progress(track_tqdm=True)):
        dir_tmp = generate_sounds(prompts, n_sounds, min_duration,
                                 max_duration, progress)
        return dir_tmp, f"Generated {n_sounds} sounds in {dir_tmp}", dir_tmp

    generate_btn.click(
        _generate, 
        inputs=[prompts_box, n_sounds, min_duration, max_duration],
        outputs=[output_dir, gen_progress, dir_input]
    )


if __name__ == "__main__":
    demo.launch(server_name="10.40.49.109")