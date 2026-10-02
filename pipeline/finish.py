#!/usr/bin/env python3
"""
finish.py: Post-render assembly for Hindi voice or read-along files.
"""
import os
import json
import wave
import subprocess
from pathlib import Path
import yaml


def seg_times(script_yaml, durations_dict=None):
    """
    Calculate segment start and duration times.
    
    Args:
        script_yaml: List of segment dicts from script.yaml
        durations_dict: Dict mapping seg_id to duration (from tts_hindi.py)
    
    Returns:
        Dict mapping seg_id to (start_time, duration)
    """
    times = {}
    start = 0.0
    
    for segment in script_yaml:
        seg_id = segment.get("id", "unknown")
        script_seconds = segment.get("seconds", 0)
        
        # If durations_dict provided, use spoken length + 0.6s fade
        if durations_dict and seg_id in durations_dict:
            spoken = durations_dict[seg_id]
            duration = max(script_seconds, spoken + 0.6)
        else:
            duration = script_seconds
        
        times[seg_id] = (start, duration)
        start += duration
    
    return times


def mix_hindi_voice(episode_dir, output_video):
    """
    Mix Hindi TTS audio onto video.
    
    Args:
        episode_dir: Path to episodes/<week> directory
        output_video: Path to final MP4 file to create
    """
    audio_dir = episode_dir / "out" / "audio"
    durations_file = episode_dir / "out" / "durations.json"
    
    # Load script and durations
    with open(episode_dir / "script.yaml") as f:
        script = yaml.safe_load(f)
    
    with open(durations_file) as f:
        durations = json.load(f)
    
    # Calculate segment timings
    times = seg_times(script, durations)
    
    # Mix audio: concatenate segment WAVs with proper timing
    # For simplicity, we create a silent base and overlay each segment
    segments = []
    for segment in script:
        seg_id = segment["id"]
        wav_file = audio_dir / f"{seg_id}.wav"
        
        if wav_file.exists():
            start_time, seg_duration = times[seg_id]
            # Add 0.3s delay to segment start (safety margin for video sync)
            adjusted_start = start_time + 0.3
            segments.append((adjusted_start, str(wav_file)))
    
    if not segments:
        # No audio files; just copy video as-is
        subprocess.run([
            "cp", 
            str(episode_dir / "out" / "video.mp4"),
            str(output_video)
        ], check=True)
        return
    
    # Create audio mixing script using ffmpeg filter_complex
    # Concatenate all WAVs, then delay them and mix into video
    filter_parts = []
    input_args = ["-i", str(episode_dir / "out" / "video.mp4")]
    
    for idx, (start_time, wav_file) in enumerate(segments):
        input_args.extend(["-i", wav_file])
        # adelay: add delay in milliseconds
        delay_ms = int(start_time * 1000)
        filter_parts.append(f"[{idx+1}]adelay={delay_ms}|{delay_ms}[a{idx}]")
    
    # Mix all delayed audio tracks
    mix_inputs = "".join([f"[a{idx}]" for idx in range(len(segments))])
    mix_filter = f"{mix_inputs}amix=inputs={len(segments)}:duration=longest[mixed]"
    filter_complex = ";".join(filter_parts) + ";" + mix_filter
    
    # Run ffmpeg to mux audio into video
    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "0:v:0",  # video from input 0
        "-map", "[mixed]",  # audio from mixed filter
        "-c:v", "copy",  # copy video codec (already h264)
        "-c:a", "aac",  # encode audio as AAC
        "-b:a", "128k",  # audio bitrate
        str(output_video)
    ]
    
    subprocess.run(cmd, check=True)
    print(f"✓ Mixed Hindi voice into {output_video}")


def create_readalong(episode_dir, lang, output_file):
    """
    Create read-along file for manual voiceover (EN/TA).
    
    Args:
        episode_dir: Path to episodes/<week> directory
        lang: Language code (en or ta)
        output_file: Path to readalong.txt file to create
    """
    with open(episode_dir / "script.yaml") as f:
        script = yaml.safe_load(f)
    
    # Use default 3-second segments for manual voiceover
    times = seg_times(script, durations_dict=None)
    
    lines = [
        f"# Read-along guide for {lang.upper()}",
        f"# Speak each line during its time window",
        ""
    ]
    
    for segment in script:
        seg_id = segment["id"]
        start, duration = times[seg_id]
        
        # Get the spoken line in the requested language
        say_lines = segment.get("say", {})
        text = say_lines.get(lang, "")
        
        if text:
            lines.append(f"[{start:.1f}s - {start+duration:.1f}s] {text}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"✓ Created read-along file: {output_file}")


def main():
    """Main post-render assembly."""
    episode_dir = Path(os.environ.get("EPISODE_DIR", "."))
    lang = os.environ.get("LANG_CODE", "en")
    
    # Ensure output directory exists
    out_dir = episode_dir / "out" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_video = out_dir / f"{episode_dir.name}_{lang}.mp4"
    
    if lang == "hi":
        # Hindi: mix TTS audio onto video
        durations_file = episode_dir / "out" / "durations.json"
        if durations_file.exists():
            mix_hindi_voice(episode_dir, output_video)
        else:
            # No durations (TTS failed or not run); copy video as-is
            subprocess.run([
                "cp",
                str(episode_dir / "out" / "video.mp4"),
                str(output_video)
            ], check=True)
            print(f"⚠ No durations found; copied silent video to {output_video}")
    else:
        # English/Tamil: copy video as-is, generate read-along
        subprocess.run([
            "cp",
            str(episode_dir / "out" / "video.mp4"),
            str(output_video)
        ], check=True)
        
        readalong_file = out_dir / f"{episode_dir.name}_{lang}_readalong.txt"
        create_readalong(episode_dir, lang, readalong_file)
    
    print(f"✓ Finished: {output_video}")


if __name__ == "__main__":
    main()
