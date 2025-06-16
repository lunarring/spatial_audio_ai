import gradio as gr
import os
import random
import string
import threading
import time
import numpy as np
import soundfile as sf
from spatial_audio_ai.generators.stable_audio import (
    StableAudioOpenSmall, SoundPoolGenerator, SpatialSoundPoolPlayer
)
from spatial_audio_ai.tools.tools import apply_fade_in_out, save_sound
from spatial_audio_ai.tools.spatializer import (
    SO_Playback, CHUNKSIZE, SAMPLING_RATE
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
    circular_loop,
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

    def playback():
        try:
            player.start_initial_sound()
            start_time = time.time()
            duration_seconds = duration_minutes * 60
            for j, chunk in enumerate(player.scene.run()):
                if playback_stop_flag:
                    print("Playback stopped by user (flag).")
                    break
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
                chunk = np.clip(chunk, -1, 1)
                player.sound_streamer.send(chunk)
                time.sleep(CHUNKSIZE/SAMPLING_RATE - 0.01)
                elapsed = time.time() - start_time
                if elapsed > duration_seconds:
                    print(f"Playback completed after {elapsed:.1f} seconds")
                    break
                if j % 430 == 0:
                    active_sounds = len(player.scene.sound_objects)
                    print(f"Time: {elapsed:.1f}s | "
                          f"Active sounds: {active_sounds}")
        except Exception as e:
            print(f"Playback error: {e}")

    playback_thread = threading.Thread(target=playback)
    playback_thread.start()
    # Simulate progress bar for duration
    for i in progress.tqdm(
        range(int(duration_minutes * 10)), desc="Playing back"
    ):
        if playback_stop_flag:
            break
        time.sleep(6)
    return f"Playback started for {duration_minutes} minutes."

def stop_playback():
    global playback_stop_flag, playback_thread
    playback_stop_flag = True
    if playback_thread and playback_thread.is_alive():
        playback_thread.join(timeout=2)
    return "Playback stopped."

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Spatial Sound Pool Generator & Player
    Generate a pool of AI sounds from prompts, then spatially play them back!
    """
    )
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

        def _generate(prompts, n_sounds, min_duration, max_duration,
                      progress=gr.Progress(track_tqdm=True)):
            dir_tmp = generate_sounds(prompts, n_sounds, min_duration,
                                      max_duration, progress)
            return dir_tmp, f"Generated {n_sounds} sounds in {dir_tmp}"

        generate_btn.click(
            _generate, inputs=[prompts_box, n_sounds, min_duration, max_duration],
            outputs=[output_dir, gen_progress]
        )

    with gr.Tab("2. Spatial Playback"):
        gr.Markdown(
            """
        Select a generated sound pool directory and play it back with spatialization.
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
        play_btn = gr.Button("Start Spatial Playback")
        stop_btn = gr.Button("Stop Playback")
        playback_status = gr.Textbox(label="Playback Status", interactive=False)

        play_btn.click(
            play_spatial_soundpool,
            inputs=[
                dir_input, p_inject, box_size, volume, duration_minutes,
                use_circular, circular_radius_min, circular_radius_max,
                circular_speed_min, circular_speed_max,
                circular_angle_min, circular_angle_max,
                circular_direction, circular_center_x, circular_center_y,
                circular_loop
            ],
            outputs=playback_status
        )
        stop_btn.click(stop_playback, outputs=playback_status)

if __name__ == "__main__":
    demo.launch() 