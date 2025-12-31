import subprocess
import json
from tqdm import tqdm
from pathlib import Path
import tempfile
from typing import Optional, List, Dict, Any
import multiprocessing
import yaml
from ttsizer.utils.logger import get_logger

logger = get_logger("audio_extractor")

class AudioExtractor:
    """Extracts audio from video files using FFmpeg based on configuration.
    
    Supports selecting preferred language audio streams and can use an intermediate
    AAC step for videos above a certain resolution threshold before final encoding.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initializes the AudioExtractor with specific configuration settings.

        Args:
            config: A dictionary containing settings for audio extraction,
                    typically from the 'audio_extractor' section of config.yaml.
        """
        self.lang_codes = config["preferred_lang_codes"]
        self.workers = config.get("max_workers")
        self.sample_rate = str(config["output_sample_rate"])
        self.codec = config["output_codec"]
        self.aac_threshold = config["resolution_threshold_for_aac_step"]
        self.aac_bitrate = config["intermediate_aac_bitrate"]
        self.ffmpeg_timeout_seconds = config.get("ffmpeg_timeout_seconds")
        self.skip_if_output_exists = bool(config.get("skip_if_output_exists", False))

    def _run_cmd(self, cmd: List[str], *, fname: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                timeout=self.ffmpeg_timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            tqdm.write(f"⏱️ FFmpeg timeout for {fname} after {self.ffmpeg_timeout_seconds}s")
            raise subprocess.CalledProcessError(returncode=-1, cmd=cmd, stderr=(e.stderr or b""))

    def _get_audio_streams(self, path: Path) -> List[Dict[str, Any]]:
        """Retrieves all audio streams from a media file using ffprobe.

        Args:
            path: Path to the media file.

        Returns:
            A list of dictionaries, where each dictionary represents an audio stream.
        """
        cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "a", str(path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
        return json.loads(result.stdout).get("streams", [])

    def _get_video_height(self, path: Path) -> Optional[int]:
        """Retrieves the video height of the first video stream in a media file.

        Args:
            path: Path to the media file.

        Returns:
            The video height as an integer, or None if not found.
        """
        cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "v:0", str(path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
        streams = json.loads(result.stdout).get("streams", [])
        return int(streams[0]["height"]) if streams and "height" in streams[0] else None

    def _get_stream_idx(self, streams: List[Dict[str, Any]]) -> Optional[int]:
        """Selects the audio stream index based on preferred language codes.

        Args:
            streams: A list of audio stream information dictionaries.

        Returns:
            The index of the preferred audio stream, or the first stream if no preference matches.
            Returns None if no audio streams are found.
        """
        for code in self.lang_codes:
            for s in streams:
                if s.get("tags", {}).get("language", "").lower() == code.lower():
                    return s.get("index")
        return streams[0].get("index") if streams else None

    def _process_file(self, in_path: Path, out_path: Path) -> bool:
        """Extracts the selected audio stream from a single video file.

        Handles an optional intermediate AAC encoding step for high-resolution videos.

        Args:
            in_path: Path to the input video file.
            out_path: Path to save the extracted audio file.

        Returns:
            True if extraction was successful, False otherwise.
        """
        if self.skip_if_output_exists and out_path.exists():
            return True

        try:
            streams = self._get_audio_streams(in_path)
            stream_idx = self._get_stream_idx(streams)
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or "").strip()
            tqdm.write(f"❌ ffprobe failed {in_path.name}: {msg[:300]}")
            return False

        if stream_idx is None:
            tqdm.write(f"❌ No audio stream in {in_path.name}")
            return False

        audio_map = f"0:{stream_idx}"
        try:
            height = self._get_video_height(in_path)
        except subprocess.CalledProcessError:
            height = None
        use_aac = bool(height and height > self.aac_threshold)

        ffmpeg_base = ["-y", "-hide_banner", "-nostats", "-nostdin", "-i", str(in_path), "-map", audio_map]
        
        try:
            if use_aac:
                with tempfile.TemporaryDirectory(prefix="tmp_") as tmp_dir:
                    tmp_path = Path(tmp_dir) / f"{in_path.stem}_tmp.aac"
                    cmd_aac = ["ffmpeg", *ffmpeg_base, "-c:a", "aac",
                             "-b:a", self.aac_bitrate, "-ar", self.sample_rate, str(tmp_path)]
                    self._run_cmd(cmd_aac, fname=in_path.name)

                    cmd_final = ["ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(tmp_path),
                               "-c:a", self.codec, "-ar", self.sample_rate, str(out_path)]
                    self._run_cmd(cmd_final, fname=in_path.name)
            else:
                cmd = ["ffmpeg", *ffmpeg_base, "-c:a", self.codec,
                      "-ar", self.sample_rate, str(out_path)]
                self._run_cmd(cmd, fname=in_path.name)
            return True
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="ignore").strip()
            last_line = err.splitlines()[-1] if err.splitlines() else 'FFmpeg error'
            tqdm.write(f"❌ Failed {in_path.name}: {last_line}")
            return False
        except Exception as e:
            tqdm.write(f"❌ Failed {in_path.name}: {type(e).__name__}: {e}")
            return False

    def _process_task(self, task: tuple[Path, Path]) -> bool:
        return self._process_file(*task)

    def process_directory(self, in_dir: Path, out_dir: Path):
        """Extracts audio from all supported video files in a directory structure.

        Uses multiprocessing to parallelize audio extraction tasks.
        Replicates the input directory structure in the output directory.

        Args:
            in_dir: Path to the root directory containing video files.
            out_dir: Path to the base directory where extracted audio will be stored.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing: {in_dir}")
        logger.info(f"Output: {out_dir}")
        
        workers = self.workers if self.workers is not None else multiprocessing.cpu_count()
        workers = max(1, workers)

        video_exts = {'.mkv', '.mp4', '.avi', '.mov', '.webm', '.wav', '.flac', '.mp3'}
        tasks = []
        for path in in_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in video_exts:
                rel_path = path.relative_to(in_dir)
                out_subdir = out_dir / rel_path.parent
                out_subdir.mkdir(parents=True, exist_ok=True)
                out_path = out_subdir / path.with_suffix(f'.{self.codec.lower()}').name
                tasks.append((path, out_path))

        if not tasks:
            logger.warning(f"No videos found in {in_dir}")
            return
    
        workers = min(workers, len(tasks))
        
        logger.info(f"Found {len(tasks)} videos. Using {workers} workers...")
        total = len(tasks)
        log_every = 1 if total <= 50 else 10

        results: List[bool] = []
        completed = 0
        success = 0

        if workers == 1:
            for task in tqdm(tasks, total=total, desc=f"Extracting: {in_dir.name}", unit="file"):
                ok = self._process_task(task)
                results.append(ok)
                completed += 1
                success += 1 if ok else 0
                if completed % log_every == 0 or completed == total:
                    logger.info(f"Progress: {completed}/{total} files (ok: {success})")
        else:
            with multiprocessing.Pool(processes=workers) as pool:
                iterator = pool.imap_unordered(self._process_task, tasks, chunksize=1)
                for ok in tqdm(iterator, total=total, desc=f"Extracting: {in_dir.name}", unit="file"):
                    results.append(ok)
                    completed += 1
                    success += 1 if ok else 0
                    if completed % log_every == 0 or completed == total:
                        logger.info(f"Progress: {completed}/{total} files (ok: {success})")

        logger.info(f"Done: {success}/{len(tasks)} files processed")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    in_dir = Path(cfg["project_setup"]["video_input_base_dir"]) / cfg["project_setup"]["series_name"]
    out_dir = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["series_name"] / cfg["audio_extractor"]["output_folder"]
    
    print(f"Extracting audio from {in_dir}")
    extractor = AudioExtractor(config=cfg["audio_extractor"])
    extractor.process_directory(in_dir, out_dir)
