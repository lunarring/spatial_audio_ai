#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import soundfile as sf
from diffusers import StableAudioPipeline
import numpy as np
import sys
sys.path.append('../../rtd_comfy/sdxl_turbo')
from embeddings_mixer import EmbeddingsMixer
import lunar_tools as lt
import torch
import os
from datetime import datetime
sys.path.append("../spatial")
from tools import AudioHandler
import random


class AudioDiffusion:
    def __init__(
        self, 
        pipe, 
        num_inference_steps=100, 
        force_mono=True,
        audio_end_in_s=10,
        auto_fade=True,
        fade_duration=0.2
    ):
        self.pipe = pipe
        self.em = EmbeddingsMixer(pipe)
        self.ah = AudioHandler(pipe.vae.sampling_rate)
        self.seed = None
        self.generator = None
        self.device = self.pipe._execution_device
        self.do_classifier_free_guidance = True
        self.num_inference_steps = num_inference_steps
        self.force_mono = force_mono
        self.audio_end_in_s = audio_end_in_s
        self.auto_fade = auto_fade
        self.fade_duration = fade_duration
        self.set_seed()

    def set_seed(self, seed=420):
        self.seed = seed
        self.generator = torch.Generator("cuda").manual_seed(seed)

    def set_random_seed(self):
        seed = np.random.randint(99999999999)
        self.set_seed(seed)

    def set_num_inference_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps

    def set_audio_end_in_s(self, audio_end_in_s):
        self.audio_end_in_s = audio_end_in_s

    def get_embedding(self, prompt):
        return self.pipe.encode_prompt(prompt, self.device, self.do_classifier_free_guidance)

    def generate_sound(self, prompt_embeds, num_waveforms_per_prompt=1):
        assert not isinstance(prompt_embeds, str), "prompt_embeds should not be a string. Use get_embedding method first."
        # prompt_embeds = self.get_embedding(prompt)
        audio = self.pipe(
            prompt_embeds=prompt_embeds,
            num_inference_steps=self.num_inference_steps,
            audio_end_in_s=self.audio_end_in_s,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            generator=self.generator,
        ).audios

        output = audio[0].T.float().cpu().numpy()
        if self.force_mono:
            output = output[:, 0]
        if self.auto_fade:
            output = self.ah.apply_fade_in_out(output, self.fade_duration)
        return output


    def get_latents(self):
        print("NOT IMPLEMENTED! LEGACY CODE")

        device = self._execution_device
        do_classifier_free_guidance = True

        p1 = self.encode_prompt(prompt1, device, do_classifier_free_guidance)


        num_channels_vae = self.transformer.config.in_channels
        waveform_length = int(self.transformer.config.sample_size)

        generator = torch.Generator("cuda").manual_seed(420)
        latents1 = self.prepare_latents(
            1 * 1,
            num_channels_vae,
            waveform_length,
            p1.dtype,
            device,
            generator,
            None,
            None,
            1,
            audio_channels=self.vae.config.audio_channels, #this is interesting
        )

    def blend_two_embeds(self, embed1, embed2, weight):
        assert 0 <= weight <= 1, "Weight should be between 0 and 1"
        blended_embed = self.em.blend_two_embeds([embed1], [embed2], weight)[0]
        
        return blended_embed




if __name__ == "__main__":
    
    # %%
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to("cuda")
    audio_diffusion = AudioDiffusion(pipe, num_inference_steps=100)
    audio_handler = AudioHandler(pipe.vae.sampling_rate)
    
    xxx
    
    # %% generate one simple sound and save 
    audio_diffusion.set_random_seed()
    audio_diffusion.set_audio_end_in_s(6)
    audio_diffusion.set_num_inference_steps(100)
    prompt = "person talking english"
    prompt_embeds = audio_diffusion.get_embedding(prompt)
    output = audio_diffusion.generate_sound(prompt_embeds)
    output = audio_handler.apply_fade_in_out(output)
    audio_handler.save_sound(output, f"talking2.wav")
    
    
    # # multichannel = np.zeros((12, 12 * 44100))
    # list_sounds = []
    # for i in range(12):
    #     list_sounds.append(output)
    #     # multichannel[i, i * 44100:(i + 1) * 44100] = output
    # audio_diffusion.save_sound_multichannel(list_sounds, "multichannel_beats")
    
    
    # %% blend two prompts
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to("cuda")
    audio_diffusion = AudioDiffusion(pipe, num_inference_steps=100)
    audio_diffusion.set_random_seed()
    prompt1 = "loud metal weird scratching"
    prompt2 = prompt1 + ", psychedelic, horrible, scary"
    nmb_sounds = 5
    
    p1 = audio_diffusion.get_embedding(prompt1)
    p2 = audio_diffusion.get_embedding(prompt2)
    
    list_sounds = []
    
    for i, weight in enumerate(np.linspace(0, 1, nmb_sounds)):
        prompt_embeds = audio_diffusion.blend_two_embeds(p1, p2, weight)[0]
        
        # run the generation using audio_diffusion class
        output = audio_diffusion.generate_sound(prompt_embeds)
        
        list_sounds.append(output)
    
        audio_handler.save_sound(output, f"sound_{i}.wav")
    
        
    #%% prompt ball
    
    
    prompt_base = "ASMR symphony"
    prompt_base = "pure rap"
    prompt_base = "tiger roar"
    
    prompt_base = "metal knocknig rusty"
    prompt_base = "a hammer hitting the nail, percursive"
    prompt_base = "percussive kick sound"
    prompt_base = "trance"
    prompt_base = "very powerful thunderstorm techno beat deep trippy"
    prompt_base = "DMT trip sound"
    prompt_base = "very powerful thunderstorm techno beat deep trippy"
    prompt_base = "palm leaves rattle rustling rhytmically loud fast, deep beat"
    
    list_augs = ["psychedelic", "beautiful", "mesmerizing", "ethereal", "new age", "tranquil", "serene", "dreamy", "hypnotic", "luminous", "celestial", "blissful", "harmonious", "radiant"]
    
    # list_augs = [
    #     "psychedelic", "trippy", "energetic", "vibrant", "intense", "hypnotic", 
    #     "pulsating", "deep", "groovy", "cosmic", "alien", "futuristic", 
    #     "mechanical", "robotic", "synthetic", "electronic", "spacey", "otherworldly", 
    #     "transcendent", "mind-bending", "euphoric", "dark", "mystical", "shamanic"
    # ]
    
    
    
    nmb_sounds = 12
    weight = 1.0 # 0.0 would be effective mono, 1.0 maximum difference between the different channels
    num_augs = 2  # more nmb_augs = more divergent sounds!
    duration = 32
    seed = 426
    num_inference_steps = 50
    
    basename = prompt_base.replace(" ", "_")
    audio_handler.set_output_dir(f"/home/lugo/audio/export/")
    prompt_embeds_base = audio_diffusion.get_embedding(prompt_base)
    audio_diffusion.set_audio_end_in_s(duration)
    audio_diffusion.set_seed(seed)
    audio_diffusion.set_num_inference_steps(num_inference_steps)
    np.random.seed(seed)  # Set the seed for numpy as well
    list_sounds = []
    new_prompts = []
    
    for i in range(nmb_sounds):
        # Randomly choose augmentations from list_augs based on num_augs
        augments = np.random.choice(list_augs, num_augs, replace=False)
        # Create the new prompt by combining the base prompt with the chosen augmentations
        new_prompt = f"{prompt_base}, {', '.join(augments)}"
        print(new_prompt)
        new_prompts.append(new_prompt)
        
        prompt_embeds_new = audio_diffusion.get_embedding(new_prompt)
        prompt_embeds_mixed = audio_diffusion.blend_two_embeds(prompt_embeds_base, prompt_embeds_new, weight)
        output = audio_diffusion.generate_sound(prompt_embeds_mixed)
        audio_handler.save_sound(output, f"sound_{i}.wav")
        list_sounds.append(output)
    
    # add last output as bass channel...
    list_sounds.append(np.mean(list_sounds, axis=0))
    # force mono
    # for i in range(nmb_sounds-1):
        # list_sounds.append(output)
    
    # Get the current date and time in the format YYMMDD_HHMM
    date_str = datetime.now().strftime("%y%m%d_%H%M")
    
    # Append the date string to the basename
    basename_with_date = f"{date_str}_{basename}"
    
    audio_handler.save_sound_multichannel(list_sounds, basename_with_date)
    audio_handler.save_sound_multichannel(list_sounds, "latest.npy")
    
    # Save prompts, settings, and seed to a text file
    settings = {
        "prompt_base": prompt_base,
        "list_augs": list_augs,
        "nmb_sounds": nmb_sounds,
        "weight": weight,
        "num_augs": num_augs,
        "duration": duration,
        "basename_with_date": basename_with_date,
        "seed": seed  
    }
    
    with open(f"/home/lugo/audio/export/{basename_with_date}.txt", "w") as f:
        f.write("Settings:\n")
        for key, value in settings.items():
            f.write(f"{key}: {value}\n")
        f.write("\nPrompts:\n")
        for prompt in new_prompts:
            f.write(f"{prompt}\n")
    
    
    #%% resave the sounds
    audio_handler = AudioHandler()
    list_sounds = []
    for i in range(13):
        sound = output.copy()
        
        # # sound = np.roll(sound, np.random.randint(50)-255)
        
        if i==8:
            list_sounds.append(sound)
        else:
        #     sound*=5
        # else:
        #     sound*=2
        #     # pass
            list_sounds.append(0*sound)
            
    audio_handler.save_sound_multichannel(list_sounds, "latest.npy", apply_rebalancing=True)
    
    #%% GENERATE A POOl
    

    class SoundPoolGenerator:
        def __init__(self, audio_handler, audio_diffusion):
            self.audio_handler = audio_handler
            self.audio_diffusion = audio_diffusion

        def generate(self, list_prompts, name_space, nmb_sounds, duration_sound = [4, 8]):
            
            num_inference_steps = 100
            self.audio_handler.set_output_dir(f'/home/lugo/audio/export/{name_space}')
            
            for _ in range(nmb_sounds):
                prompt = random.choice(list_prompts)
                duration = random.uniform(duration_sound[0], duration_sound[1])
                seed = np.random.randint(9999999)

                self.audio_diffusion.set_seed(seed)
                self.audio_diffusion.set_audio_end_in_s(duration)
                self.audio_diffusion.set_num_inference_steps(num_inference_steps)

                prompt_embeds = self.audio_diffusion.get_embedding(prompt)
                sound = self.audio_diffusion.generate_sound(prompt_embeds)
                sound = self.audio_handler.apply_fade_in_out(sound)

                filename = prompt.replace(" ", "_")
                filename = filename.replace("'", "")
                filename = filename.replace('"', "")
                filename = filename.replace('.', "")
                self.audio_handler.save_sound(sound, f"{filename}_{seed}.wav")
                print(f"Sound generation complete. File '{filename}_{seed}.wav' saved successfully.")

    
    list_prompts = [
    "leaves rustling", 
    "wind blowing tree", 
    "tree moving", 
    "wind howling", 
    "owl singing", 
    "bird singing",
    "branches creaking", 
    "footsteps crunching on twigs", 
    "river gently flowing", 
    "woodpecker tapping", 
    "frogs croaking near the water", 
    "insects buzzing in the air", 
    "distant waterfall splashing", 
    "deer cautiously stepping on the grass", 
    "squirrels chattering in the trees", 
    "distant thunder rumbling", 
    "rain softly pattering on leaves", 
    "crickets chirping in the twilight", 
    "branches snapping underfoot", 
    "wolves howling in the distance", 
    "leaves crunching underfoot", 
    "the faint hum of the breeze", 
    "moss squishing under boots", 
    "the echo of a distant woodpecker", 
    "pine needles falling softly", 
    "the gentle crackle of a campfire", 
    "birds fluttering through the canopy", 
    "a stream babbling over rocks", 
    "trees creaking in the wind", 
    "owls hooting softly", 
    "leaves swirling in a gust", 
    "water droplets falling from branches", 
    "a bear rustling through the underbrush", 
    "wild boar snorting in the bushes", 
    "the soft thud of acorns dropping", 
    "tree frogs calling from the shadows", 
    "the far-off cry of an eagle", 
    "branches rubbing against each other", 
    "wind whistling through hollow logs", 
    "the gurgle of a hidden spring", 
    "foxes yipping at dawn", 
    "bat wings fluttering overhead", 
    "the snap of dry twigs in the wind", 
    "an elk bugling in the distance", 
    "crows cawing as they fly by", 
    "the squawk of a distant jay", 
    "wild turkeys rustling through the undergrowth", 
    "the faint rustle of an animal burrowing", 
    "ants marching in the grass", 
    "branches bouncing after a bird takes flight"
]

    list_prompts = [
        "glorp beast bubbling under the moss", 
        "squelching sound of a fizzle-toad hopping", 
        "wing-flutter of a glowing dusk-moth", 
        "chirruping giggle of a shadow sprite", 
        "whirring hum of a tree-dwelling zorn", 
        "crackle-popping of a flame-tail wisp", 
        "slurping echoes of a tunnel-dweller", 
        "raspy hiss of the thorn-winged wyrl", 
        "metallic chime of a crystal-feathered owl", 
        "guttural grunts of a mud-lurking groblin", 
        "warbling hum from a canopy-dwelling nightshade bird", 
        "thumping tail of a boulder-back tortifant", 
        "sizzling screech of a lightning-scaled zeth", 
        "snapping click of a spore-spraying jorv", 
        "rhythmic tapping of the hollow-boned clinkbat", 
        "hollow echo of a deepwood banshee’s wail", 
        "skittering scrape of the four-legged bark crawler", 
        "high-pitched shrill of a glass-winged krilf", 
        "low rumble of the moss-dwelling earthhound", 
        "splashing footsteps of the bog-dwelling finchuck", 
        "metallic creak of a rust-scaled drakeling", 
        "whistling trill of the midnight flickerbat", 
        "gloopy growl of a sludge-thrasher", 
        "whispering chirp of a mist-whisp winder", 
        "subtle pulse of a lantern-eyed nythe", 
        "buzzing hum of a nectar-seeking farnfly", 
        "clattering clink of the stone-legged gryth", 
        "spinning whoosh of the spiral-tailed flindle", 
        "echoing yawn of a rock-mouth deepstalker", 
        "crunching sound of a vine-wrapped brem", 
        "fizzing pop of an acid-spitting vreeg", 
        "gurgling croak of a shadow-dwelling glibber", 
        "squeaky song of the sky-swimming flitterfish", 
        "chirping purr of a leaf-mimic spindler", 
        "drumming tap of a hollow tree-beast", 
        "snuffling snort of a root-sniffing blorft", 
        "scraping shuffle of a bark-skinned skrall", 
        "electric zap of a lightning-bug krund", 
        "crackling hiss of a frost-breathing wyrmlet", 
        "throaty buzz of the horn-winged krez", 
        "rustling roll of the tumblevine creeper", 
        "tick-tick-tick of the time-skipping clockbeetle", 
        "whispering chant of the forest-bound fog weaver", 
        "fluttering whoop of a sky-bound gloamshriek", 
        "stomping thud of a ground-pounding rootback", 
        "chirping clicks of the spider-like leafhound", 
        "booming echo of a cave-dwelling thundercrest", 
        "slithering hiss of the vine-covered snarltrap", 
        "vibrating hum of the prism-winged duskshrike"
    ]
    
    list_prompts = [
        "twinkling chimes of floating fairy dust", 
        "melodic hum of a glowing moonflower", 
        "soft tinkling of crystal dew drops", 
        "whispering song of the wind through enchanted trees", 
        "gentle sigh of a sleeping dream fox", 
        "fluttering wings of shimmering pixies", 
        "sparkling crackle of a spellbinding charm", 
        "echoing giggles from invisible forest spirits", 
        "bells chiming in the distant air", 
        "faint harp strings strummed by invisible hands", 
        "delicate ripple of time-flowing water", 
        "soft laughter of dancing starlight wisps", 
        "bubbling glow of a magic cauldron", 
        "shimmering hum of a mystic mirror", 
        "humming pulse of an enchanted gemstone", 
        "gentle tapping of tiny elf feet on mushroom tops", 
        "whirling swirls of an invisible wind dancer", 
        "tinkling rain of rainbow-colored sparkles", 
        "hollow echo of a faraway magical bell", 
        "mysterious rustle of sentient leaves", 
        "floating notes of an unseen lullaby", 
        "melting sound of a liquid crystal stream", 
        "soft plucking of enchanted harp vines", 
        "whispered riddles carried by the breeze", 
        "distant glow of stardust falling on leaves", 
        "glittering chime of a fairy queen’s laughter", 
        "crackling shimmer of enchanted fireflies", 
        "echoing whispers of ancient forest spirits", 
        "bubbling giggle of mischievous sprites", 
        "flitting sparkle of a twilight wisp", 
        "whistling echo of a moonlit waterfall", 
        "crystalline ringing of a magical key turning", 
        "soft whoosh of a spell unfolding in the air", 
        "distant flute notes carried by glowing mist", 
        "tinkling dance of stardust underfoot", 
        "whirling hum of a time-bending portal", 
        "faint purr of a glowing Cheshire cat", 
        "lilting echo of laughter from unseen realms", 
        "mystical murmur of ancient stones", 
        "twilight shimmer of a glowing night blossom", 
        "glistening trickle of a silver moonwell", 
        "subtle hum of magical vines weaving", 
        "gentle ringing of enchanted silver bells", 
        "sparkling crackle of a rainbow arc forming", 
        "whispered secrets of a dreamweaver’s spell", 
        "hollow call of a faraway stardancer", 
        "mystic vibrations of an ethereal harp", 
        "soft gliding of a moon-glow swan", 
        "fluttering pulse of enchanted butterfly wings", 
        "dreamy hum of a floating cloud castle", 
        "fizzing whisper of a potion swirling", 
        "symphonic rustle of a thousand enchanted leaves"
    ]
    
    list_prompts = [
    "low, throaty growl from deep within", 
    "soft, breathy sigh escaping parted lips", 
    "rhythmic panting in the heat of the moment", 
    "gentle moan carried by the breeze", 
    "animalistic grunt of exertion", 
    "subtle rustling of bodies moving together", 
    "the wet sound of lips softly parting", 
    "deep, rumbling purr of satisfaction", 
    "sharp intake of breath just before a kiss", 
    "slow, deliberate exhale in the silence", 
    "soft whimper from a moment of pleasure", 
    "low, vibrating hum deep in the chest", 
    "the slap of skin meeting skin", 
    "quiet, primal growl during a close encounter", 
    "the sharp crack of a whip in the air", 
    "heavy, slow breathing in the dark", 
    "the gentle scrape of nails against skin", 
    "sudden gasp of surprise and desire", 
    "the wet sound of a tongue exploring", 
    "purring vibrations of contentment and desire", 
    "deep, guttural groan from the core", 
    "quickening breaths of anticipation", 
    "snarling bite just beneath the surface", 
    "the wet slide of lips over skin", 
    "hushed whispers of passion in the dark", 
    "urgent, hungry growling in the heat", 
    "slow, rhythmic breathing in sync", 
    "the primal roar of a predator claiming its own", 
    "soft, whispered sighs of pleasure", 
    "quick, heated exhales in the moment", 
    "animalistic snarl of dominance", 
    "the slow scrape of teeth over flesh", 
    "trembling moan in the midst of passion", 
    "the rough, wet sound of lips pressing together", 
    "deep, slow rumble of an animal asserting control", 
    "the sharp, sensual slap of a tail against the ground", 
    "heated exhale that escapes with a shudder", 
    "feral grunt of satisfaction", 
    "soft lick of a tongue against warm skin", 
    "whispered growls in the night air", 
    "breathy moan carried by the wind", 
    "the primal call of a beast in heat", 
    "the wet sound of a kiss echoing in the quiet", 
    "deep, husky groans of desire", 
    "snorting breaths of an animal during mating", 
    "delicate whimpers beneath the surface", 
    "the shiver-inducing scrape of fur against skin", 
    "growling exhale as bodies connect", 
    "trembling sigh of release", 
    "the heavy thud of hooves in heated pursuit", 
    "raw, urgent gasps in the wild", 
    "low, seductive purr vibrating through the night"
    
    ]
    
    list_prompts = [
        "soft, decadent moan of satisfaction", 
        "deliciously slow exhale after a deep sip of wine", 
        "breathy giggle trailing off into a sigh", 
        "low, throaty hum of pure indulgence", 
        "the subtle sound of lips brushing together", 
        "sensuous murmur in the dark", 
        "silky whisper of words against skin", 
        "languid moan escaping during bliss", 
        "slow, luxurious breath after tasting something sweet", 
        "the soft smack of lips after a kiss", 
        "gentle, teasing laugh from lips barely apart", 
        "deep inhale of scented air before sinking back into pillows", 
        "wet, teasing flick of a tongue against lips", 
        "sigh of delight as fingers graze bare skin", 
        "the quiet, intimate sound of hands caressing", 
        "hushed gasp followed by a lazy smile", 
        "slow, rhythmic breathing intertwined with soft laughter", 
        "sated, contented moan in the afterglow", 
        "the low purr of pleasure vibrating in the throat", 
        "playful hum as lips curl into a mischievous grin", 
        "soft, lingering kiss that melts into silence", 
        "quiet, intoxicating murmur in a lover's ear", 
        "gentle, indulgent lick of the lips", 
        "whispered moan of pleasure shared in secret", 
        "delightful sigh of surrender under soft sheets", 
        "luxuriant hum of contentment after a sip of champagne", 
        "wet, decadent smack of lips during a kiss", 
        "faint, shivering gasp just before a kiss lands", 
        "lazy, sensual purr of contentment while stretching", 
        "low, sultry groan of indulgence", 
        "the slow, satisfying sound of breath being released", 
        "a teasing giggle whispered in close quarters", 
        "tender hum of enjoyment as fingers trace skin", 
        "sated, deep exhale after an indulgent bite", 
        "the soft rustle of silk sheets as bodies shift", 
        "gentle murmur of approval as lips press to skin", 
        "deliberate, slow sigh of pure hedonistic bliss", 
        "the faint sound of teeth gently nibbling on a lower lip", 
        "whispered breath cascading across a collarbone", 
        "the rich sound of a body sinking into plush cushions", 
        "playful hum against the pulse point of a neck", 
        "a deep, teasing inhale just before a kiss", 
        "the lightest gasp of surprise mixed with delight", 
        "low, indulgent chuckle shared between lovers", 
        "subtle moan in response to a soft touch", 
        "hushed laughter dripping with temptation", 
        "the slow, wet sound of a lingering kiss", 
        "deep, contented exhale as fingers trace down a back", 
        "the intimate rustle of clothing slowly being undone", 
        "sensuous murmur of delight between parted lips"
    ]
    
    list_prompts = [
    "slow, rhythmic drumbeats pulsing with raw energy", 
    "soft rattle of enchanted shells in sync with swaying hips", 
    "low, sensual hum rising in rhythmic waves", 
    "mysterious whispers laced with primal desire", 
    "deep, throaty chant carried by the wind in pulses", 
    "the hypnotic rustle of feathers brushing against skin", 
    "sharp clinking of metal anklets in a seductive rhythm", 
    "soft, breathy exhale mixing with rhythmic drumming", 
    "the teasing rattle of bones in unpredictable patterns", 
    "vibrating hum of crystal bowls pulsing through the air", 
    "sensual moan harmonized with the soft shake of a rattle", 
    "echoing claps of hands in a primal, intimate rhythm", 
    "swaying chimes mimicking the rise and fall of breath", 
    "sharp, unpredictable snaps of fingers summoning magic", 
    "pulsing drumbeats that grow in intensity, then fade", 
    "the wet slide of fingers over a taut drumskin", 
    "low, guttural growl interspersed with heavy breathing", 
    "the deep, erotic hum of a didgeridoo vibrating through the night", 
    "soft, sensual giggles woven into the beat of sacred drums", 
    "echoing whispers of desire layered over rhythmic chants", 
    "clicking of beads in sync with primal movement", 
    "the unpredictable rattle of a ceremonial staff tapping in rhythm", 
    "breathy sighs caught on the wind, dancing between drumbeats", 
    "sharp, playful snaps of leather strings against bare skin", 
    "deep, primal chant followed by a sudden silence", 
    "the sensual rhythm of footsteps moving through soft earth", 
    "crackling of fire mixed with the soft murmur of intimate words", 
    "moans of pleasure syncing with the rise and fall of drums", 
    "soft, rhythmic claps of hands and bodies moving together", 
    "tinkling bells that shimmer in irregular, teasing bursts", 
    "the deep, rumbling purr of a drum vibrating with desire", 
    "faint gasps of breath layering over primal percussion", 
    "low, sultry chant interwoven with sharp percussion beats", 
    "the slow, rhythmic brush of fingers against a wooden drum", 
    "languid, sensual hum of wind chimes swaying unpredictably", 
    "sharp, echoing crack of a whip followed by a rhythmic drumbeat", 
    "rising and falling breath synced with primal shamanic rhythms", 
    "deep, vibrating hum that crescendos then drops suddenly", 
    "the soft rustle of fabric moving with rhythmic energy", 
    "sharp, teasing laughter woven into the beat of drums", 
    "low, primal moan harmonizing with the crackling of fire", 
    "soft hissing breaths syncing with the beat of enchanted rattles", 
    "deep, resonant call of a horn blending with rhythmic stomping", 
    "sharp clicking of stones against one another in an intimate pattern", 
    "faint, erratic pulse of whispered chants carried by the wind", 
    "the wet, sensual slide of hands against drum leather", 
    "mystical sighs mixed with unpredictable bursts of percussion", 
    "echoing hums laced with desire as they rise and fall", 
    "sensual gasps interspersed with deep, primal drumbeats", 
    "sharp intake of breath punctuated by a soft rattle", 
    "faint laughter carried on the wind, swirling around rhythmic chants"
    ]
    
    list_prompts = [
        "deep, bassy 'kick drum' made by a sharp 'b' sound from the lips", 
        "tight, crisp 'hi-hat' created with rapid 'tss' tongue clicks", 
        "sharp 'snare' snap using a quick 'pff' or 'k' burst from the mouth", 
        "rolling 'bass drop' produced by deep throat vibrations and humming", 
        "percussive 'clap' sound made by a hollow 'pah' with cupped hands over mouth", 
        "vibrating 'cymbal crash' recreated with a long, breathy 'shh' sound", 
        "pulsating 'bass kick' sound made by pressing lips together and exhaling quickly", 
        "soft 'rimshot' tap mimicked by a gentle tongue-click on the roof of the mouth", 
        "rapid-fire 'hi-hat' hits with short 'ts-ts-ts' bursts between breaths", 
        "thumping 'bass' rumble generated by pressing the tongue against the throat", 
        "slapping 'snare' created by pushing air forcefully between lips and teeth", 
        "sizzling 'open hi-hat' effect made by dragging out a 'sss' sound", 
        "choppy 'tom drum' rolls produced with quick alternating 'b' and 'd' sounds", 
        "vibrating 'bass wobble' using throat rumble and humming control", 
        "echoing 'kick-snare' combo with a heavy 'buh' for the kick and sharp 'psh' for the snare", 
        "sudden 'cymbal crash' recreated by a high-pitched, long 'tsisssh' with air expelled through teeth", 
        "grinding 'scratch' effect made by dragging a 'krshh' sound while tapping the tongue", 
        "repeating 'drum roll' achieved by rapid 'brrr' lip vibrations", 
        "resonant 'sub-bass' made by humming low while puffing cheeks", 
        "snappy 'snare roll' produced with quick bursts of 'tk-tk-tk'", 
        "vocal 'hi-hat shuffle' created with alternating 't-t-t' and 's-s-s' sounds", 
        "sharp 'pop' from the mouth mimicking a finger snap", 
        "heavy 'kick drum' using a deep chesty 'buh' sound with air pressure", 
        "choking 'cymbal' effect made by dragging out an airy 'chhhh' sound", 
        "pulsing 'bass drum' effect created with diaphragm control for deep 'oomph' sounds", 
        "quick 'snare hit' with a popping 'psht' sound from the back of the mouth", 
        "breathy 'rimshot' effect using a whispered 'fuh' sound", 
        "clicking 'tongue snare' by creating a 'tk' noise with the tip of the tongue", 
        "long, vibrating 'wobble bass' produced by humming deeply with the lips pressed", 
        "scratch-like sound by creating a sharp, jagged 'chk' with rapid tongue movements", 
        "staggered 'kick-snare' combination with alternating 'buh' and 'psh' sounds", 
        "sharp 'snare drum' with a forceful 'pk' sound pushed from the back of the throat", 
        "clicking 'hi-hat' imitated by quick, tight 'ts-ts' sounds made with the tongue", 
        "booming 'sub-bass' recreated by deep, sustained humming with chest resonance", 
        "stuttering 'kick drum' effect made by repeating 'buh-buh-buh' in short bursts", 
        "whispered 'shaker' sound made by breathing quickly through the teeth", 
        "glitchy 'snare hit' using a staccato 'krrk' sound with a quick tongue slap", 
        "fading 'cymbal crash' with a drawn-out 'shhhhh' sound fading into silence", 
        "repeating 'bass drum' rolls using rapid 'b-b-b' bursts with forceful air", 
        "deep, rolling 'thunder kick' made by humming through vibrating lips", 
        "high-pitched 'cymbal tap' using a soft, crisp 't-t-t' from the tongue", 
        "sharp 'snare break' with a sudden, forceful 'pshht' expelled from the back of the mouth", 
        "vibrating 'bass hum' made by holding a deep tone with lips buzzing", 
        "quick 'drum fill' mimicked with alternating 'buh' and 'tuh' beats", 
        "whirring 'scratch' sound by dragging a high-pitched 'krk-krk' with rapid tongue movements", 
        "tapping 'cymbal ride' using a soft 't-t-t' overlaid with a light 'ssshhh'"
    ]
    
    list_prompts2 = [
    "deep, resonant roars of a jaguar echoing through the trees", 
    "sharp squawks of brightly colored macaws flying overhead", 
    "rhythmic drumming from distant indigenous ceremonies", 
    "guttural growls of howler monkeys reverberating through the canopy", 
    "constant buzzing and hum of insects surrounding the explorers", 
    "the soft patter of raindrops falling onto broad jungle leaves", 
    "chanting voices of tribal people carried on the wind", 
    "low, throaty calls of toucans echoing from hidden perches", 
    "the rustling of leaves as unseen creatures move through the underbrush", 
    "sharp clicks and whistles of dolphins swimming in the river", 
    "the deep, primal beat of a drum, marking the pulse of a distant ritual", 
    "chirping calls of tree frogs resonating in the moist air", 
    "intermittent birdcalls ringing through the dense foliage", 
    "sharp hisses of snakes slithering quietly across the forest floor", 
    "soft rustling of palm fronds as the wind moves through the canopy", 
    "the steady thud of footsteps on soft, mossy ground", 
    "crackling firewood in the camp of local tribes", 
    "clicking and chattering of capuchin monkeys high in the trees", 
    "low murmur of water flowing gently through jungle streams", 
    "the haunting wail of a distant bird of prey", 
    "soft chants mixed with flute music floating from a hidden village", 
    "snapping of branches as large animals pass nearby", 
    "sporadic whooping calls of distant tribal celebrations", 
    "buzzing of mosquitoes swarming in the humid air", 
    "fluttering of bat wings as they dart between branches", 
    "the rumbling growl of a caiman sliding into the river", 
    "sharp rhythmic hand claps during tribal dances", 
    "hollow thuds of wooden sticks being struck together in ceremony", 
    "deep, rumbling thunder in the distance, signaling an approaching storm", 
    "the gentle sway of water as canoes glide silently along the river", 
    "the steady, hypnotic beat of a tribal drum deep within the jungle", 
    "sudden bursts of laughter from tribal children playing in the village", 
    "soft chanting in an unknown language, accompanied by flute notes", 
    "the harsh cawing of vultures circling high above", 
    "the clatter of falling fruits and nuts, disturbed by animals overhead", 
    "echoing roars of territorial jaguars deep in the jungle", 
    "the fast pattering of raindrops on broad, leathery leaves", 
    "rhythmic stomping of feet during a ritualistic dance", 
    "quiet splashes from river otters swimming in a nearby stream", 
    "high-pitched screech of a harpy eagle hunting from above", 
    "constant rustle of the jungle as it buzzes with life", 
    "steady, whispered chants of elders during sacred ceremonies", 
    "the distant beat of hand-carved drums, growing louder with every step", 
    "sharp crackle of twigs breaking underfoot", 
    "low, warbling sounds of a tapir moving through the dense undergrowth", 
    "the steady hum of cicadas, building in intensity as night falls", 
    "the distant clamor of tribal voices during communal gatherings", 
    "soft croaks of frogs harmonizing with the rhythm of nature", 
    "high-pitched chirps of crickets filling the evening air", 
    "sharp, sudden cry of a startled animal darting through the trees", 
    "steady clapping of hands in rhythm with chanting voices", 
    "low growl of a wild boar foraging nearby", 
    "mysterious rustling from unseen creatures scurrying across the forest floor"
]

list_prompts = [
    "gentle lapping of small waves against the shoreline", 
    "rhythmic crashing of powerful waves onto rocky cliffs", 
    "soft swish of water receding back into the sea", 
    "deep rumble of distant waves rolling across the horizon", 
    "frothy hiss of foam as the waves break on the sand", 
    "steady pulse of waves gently rocking a boat", 
    "sharp slap of water against the hull of a ship", 
    "muffled roar of waves heard underwater", 
    "whispering splash of small waves around submerged rocks", 
    "violent crash of storm-driven waves smashing into the coast", 
    "slow, deep surge of a wave building before breaking", 
    "soft bubbling sound as seawater fills tide pools", 
    "rolling hiss of pebbles being dragged by retreating waves", 
    "sharp splash of a wave cresting against a pier", 
    "steady thrum of waves pounding against a distant reef", 
    "the quiet lap of water gently kissing the shoreline at dusk", 
    "thunderous boom of a large wave hitting a rock formation", 
    "hollow echo of waves crashing inside a sea cave", 
    "the rhythmic whoosh of water pulled back by the tides", 
    "soothing pulse of waves gently breaking under a pier", 
    "constant background murmur of waves on a quiet beach", 
    "the hiss of waves melting into the sand after a storm", 
    "fizzing sound of surf breaking onto coral reefs", 
    "the low growl of waves churning against each other", 
    "sharp splash of a wave slamming into a sea wall", 
    "splashing sound of water cascading over tide-worn rocks", 
    "rising and falling roar of waves during high tide", 
    "the faintest whisper of ripples in a shallow lagoon", 
    "crackling sound of water sweeping over seashells", 
    "hollow slosh of water trapped between boulders", 
    "sharp sizzle of a retreating wave over sun-warmed sand", 
    "slow, methodical thump of waves against a dock", 
    "tumbling splash of water breaking against driftwood", 
    "faint gurgle of seawater trickling through rocks", 
    "the rolling surge of a crashing wave growing louder", 
    "steady roar of the surf as it builds before high tide", 
    "quiet bubbling as waves flow over submerged seaweed", 
    "the snapping pop of waves meeting a jagged rock", 
    "rhythmic pulse of incoming waves on a moonlit night", 
    "the low, distant roar of waves during a calm storm", 
    "hissing sound of seawater fizzing over a sandy shore", 
    "sharp splash of a wave cutting through shallow water", 
    "the thunderous boom of ocean waves hitting a reef wall", 
    "faint gurgle of water trickling through rocky crevices", 
    "rising hiss of water rushing up the beach before retreating", 
    "splashing sound of a large wave crashing into a narrow cove", 
    "constant roar of surf under a stormy sky", 
    "sharp fizzing of waves receding through pebble"]
    
    
    spg = SoundPoolGenerator(audio_handler, audio_diffusion)
    spg.generate(list_prompts, "ocean", 500, duration_sound=[2,8])

    # name_space = "forest"
    # duration_sound = [4, 8]
    # num_inference_steps = 100
    # audio_handler.set_output_dir(f'/home/lugo/audio/export/{name_space}')
    
    # while True:
    #     prompt = random.choice(list_prompts)
    #     duration = random.uniform(duration_sound[0], duration_sound[1])
    #     seed = np.random.randint(9999999)

    #     audio_diffusion.set_seed(seed)
    #     audio_diffusion.set_audio_end_in_s(duration)
    #     audio_diffusion.set_num_inference_steps(num_inference_steps)

    #     prompt_embeds = audio_diffusion.get_embedding(prompt)
    #     sound = audio_diffusion.generate_sound(prompt_embeds)
    #     sound = audio_handler.apply_fade_in_out(sound)

    #     filename = prompt.replace(" ", "_")
    #     filename = filename.replace("'", "")
    #     filename = filename.replace('"', "")
    #     filename = filename.replace('.', "")
    #     audio_handler.save_sound(sound, f"{filename}_{seed}.wav")
    #     print(f"Sound generation complete. File '{filename}.wav' saved successfully.")



