#!/usr/bin/env python3
"""
Extended YouTube dog bark audio downloader.
Runs for 10 hours, downloading audio from multiple search queries per breed.
"""

import subprocess
import os
import json
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

# Top 100 dog breeds
DOG_BREEDS = [
    "French Bulldog",
    "Labrador Retriever",
    "Golden Retriever",
    "German Shepherd Dog",
    "Poodle",
    "Dachshund",
    "Beagle",
    "Rottweiler",
    "Bulldog",
    "German Shorthaired Pointer",
    "Yorkshire Terrier",
    "Australian Shepherd",
    "Cavalier King Charles Spaniel",
    "Cane Corso",
    "Pembroke Welsh Corgi",
    "Doberman Pinscher",
    "Boxer",
    "Miniature Schnauzer",
    "Bernese Mountain Dog",
    "Shih Tzu",
    "Great Dane",
    "Pomeranian",
    "Boston Terrier",
    "Miniature American Shepherd",
    "Havanese",
    "Siberian Husky",
    "Chihuahua",
    "English Springer Spaniel",
    "Shetland Sheepdog",
    "Border Collie",
    "Brittany",
    "Cocker Spaniel",
    "Belgian Malinois",
    "Basset Hound",
    "Vizsla",
    "English Cocker Spaniel",
    "Maltese",
    "Pug",
    "Collie",
    "Mastiff",
    "West Highland White Terrier",
    "Shiba Inu",
    "Rhodesian Ridgeback",
    "Papillon",
    "Portuguese Water Dog",
    "Bichon Frise",
    "Newfoundland",
    "Dalmatian",
    "Australian Cattle Dog",
    "Whippet",
    "Wirehaired Pointing Griffon",
    "Chesapeake Bay Retriever",
    "Samoyed",
    "Scottish Terrier",
    "Weimaraner",
    "Italian Greyhound",
    "Giant Schnauzer",
    "Bloodhound",
    "Akita",
    "Cardigan Welsh Corgi",
    "German Wirehaired Pointer",
    "Saint Bernard",
    "Staffordshire Bull Terrier",
    "Airedale Terrier",
    "Boykin Spaniel",
    "Russell Terrier",
    "Bullmastiff",
    "Chinese Crested",
    "Nova Scotia Duck Tolling Retriever",
    "Biewer Terrier",
    "Cairn Terrier",
    "Bull Terrier",
    "Miniature Pinscher",
    "Great Pyrenees",
    "Soft Coated Wheaten Terrier",
    "Irish Wolfhound",
    "American Staffordshire Terrier",
    "Lagotto Romagnolo",
    "Alaskan Malamute",
    "Basenji",
    "Rat Terrier",
    "Great Swiss Mountain Dog",
    "Chinese Shar-Pei",
    "Brussels Griffon",
    "Irish Setter",
    "Anatolian Shepherd Dog",
    "Pekingese",
    "Chow Chow",
    "Old English Sheepdog",
    "Keeshond",
    "Standard Schnauzer",
    "Coton de Tulear",
    "English Setter",
    "Lhasa Apso",
    "Dogo Argentino",
    "Dogue de Bordeaux",
    "Flat-Coated Retriever",
    "Border Terrier",
    "Leonberger",
    "Beauceron",
]

# Search query templates - will be combined with breed names
SEARCH_TEMPLATES = [
    "{breed} barking",
    "{breed} bark",
    "{breed} barking sound",
    "{breed} howling",
    "{breed} growling",
    "{breed} dog barking",
    "{breed} puppy barking",
    "{breed} angry barking",
    "{breed} loud bark",
    "{breed} dog sounds",
    "{breed} vocalization",
    "{breed} whining",
    "{breed} aggressive bark",
    "{breed} guard dog barking",
    "{breed} alert barking",
]

OUTPUT_DIR = Path("/Users/dillon_davis/airlab/dog_bark_audio/youtube")
VIDEOS_PER_SEARCH = 15  # Videos per search query
RUN_DURATION_HOURS = 10

# Track what we've already downloaded to avoid duplicates
downloaded_ids = set()


def sanitize_dirname(name: str) -> str:
    """Convert breed name to a valid directory name."""
    return name.lower().replace(" ", "_").replace("-", "_").replace("'", "")


def load_existing_downloads():
    """Load IDs of already downloaded videos."""
    global downloaded_ids
    for mp3_file in OUTPUT_DIR.rglob("*.mp3"):
        # Extract video ID from filename (format: VIDEOID_title.mp3)
        video_id = mp3_file.stem.split("_")[0]
        if len(video_id) == 11:  # YouTube IDs are 11 chars
            downloaded_ids.add(video_id)

    # Also check webm files in case conversion failed
    for webm_file in OUTPUT_DIR.rglob("*.webm"):
        video_id = webm_file.stem.split("_")[0]
        if len(video_id) == 11:
            downloaded_ids.add(video_id)

    print(f"Found {len(downloaded_ids)} existing downloads")


def convert_webm_to_mp3():
    """Convert any remaining webm files to mp3."""
    webm_files = list(OUTPUT_DIR.rglob("*.webm"))
    if webm_files:
        print(f"\nConverting {len(webm_files)} webm files to mp3...")
        for webm_file in webm_files:
            mp3_file = webm_file.with_suffix(".mp3")
            if not mp3_file.exists():
                try:
                    subprocess.run(
                        ["ffmpeg", "-i", str(webm_file), "-vn", "-acodec", "libmp3lame",
                         "-q:a", "2", str(mp3_file), "-y", "-loglevel", "error"],
                        timeout=60,
                        capture_output=True
                    )
                    if mp3_file.exists():
                        webm_file.unlink()  # Remove webm after successful conversion
                except Exception as e:
                    pass  # Keep webm if conversion fails


def download_search_query(breed: str, search_query: str, breed_dir: Path) -> int:
    """Download audio for a specific search query."""
    global downloaded_ids

    # Create archive file to track downloads for this session
    archive_file = breed_dir / ".downloaded_ids.txt"

    cmd = [
        "yt-dlp",
        f"ytsearch{VIDEOS_PER_SEARCH}:{search_query}",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "192K",
        "-o", str(breed_dir / "%(id)s_%(title).50s.%(ext)s"),
        "--no-playlist",
        "--ignore-errors",
        "--no-warnings",
        "--quiet",
        "--no-progress",
        "--match-filter", "duration < 600",  # Max 10 min videos
        "--match-filter", "duration > 3",    # Min 3 sec videos
        "--restrict-filenames",
        "--download-archive", str(archive_file),
        "--sleep-interval", "1",
        "--max-sleep-interval", "3",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout per search
        )

        # Count new mp3 files
        current_mp3s = set(f.stem.split("_")[0] for f in breed_dir.glob("*.mp3"))
        new_downloads = len(current_mp3s - downloaded_ids)
        downloaded_ids.update(current_mp3s)

        return new_downloads

    except subprocess.TimeoutExpired:
        return 0
    except Exception as e:
        return 0


def download_breed_round(breed: str, template_idx: int) -> dict:
    """Download audio for one breed using one search template."""
    breed_dir = OUTPUT_DIR / sanitize_dirname(breed)
    breed_dir.mkdir(parents=True, exist_ok=True)

    template = SEARCH_TEMPLATES[template_idx % len(SEARCH_TEMPLATES)]
    search_query = template.format(breed=breed)

    count = download_search_query(breed, search_query, breed_dir)

    return {
        "breed": breed,
        "query": search_query,
        "downloaded": count,
    }


def get_breed_file_count(breed: str) -> int:
    """Get number of files for a breed."""
    breed_dir = OUTPUT_DIR / sanitize_dirname(breed)
    if breed_dir.exists():
        return len(list(breed_dir.glob("*.mp3"))) + len(list(breed_dir.glob("*.webm")))
    return 0


def main():
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=RUN_DURATION_HOURS)

    print("=" * 70)
    print("EXTENDED YouTube Dog Bark Audio Downloader")
    print("=" * 70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:   {RUN_DURATION_HOURS} hours")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Breeds:     {len(DOG_BREEDS)}")
    print(f"Search templates: {len(SEARCH_TEMPLATES)}")
    print(f"Videos per search: {VIDEOS_PER_SEARCH}")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing downloads
    load_existing_downloads()

    # Stats tracking
    stats = {
        "start_time": start_time.isoformat(),
        "total_downloaded": 0,
        "rounds_completed": 0,
        "by_breed": {breed: 0 for breed in DOG_BREEDS},
    }

    round_num = 0

    while datetime.now() < end_time:
        round_num += 1
        round_start = datetime.now()
        time_remaining = end_time - datetime.now()
        hours_left = time_remaining.total_seconds() / 3600

        print(f"\n{'='*70}")
        print(f"ROUND {round_num} | Time remaining: {hours_left:.1f} hours")
        print(f"{'='*70}")

        # Shuffle breeds for variety
        breeds_this_round = DOG_BREEDS.copy()
        random.shuffle(breeds_this_round)

        # Use different search template each round
        template_idx = (round_num - 1) % len(SEARCH_TEMPLATES)
        template = SEARCH_TEMPLATES[template_idx]
        print(f"Search template: '{template}'")
        print("-" * 70)

        round_downloads = 0

        for i, breed in enumerate(breeds_this_round, 1):
            if datetime.now() >= end_time:
                print("\nTime limit reached!")
                break

            # Prioritize breeds with fewer downloads
            current_count = get_breed_file_count(breed)

            result = download_breed_round(breed, template_idx)

            if result["downloaded"] > 0:
                round_downloads += result["downloaded"]
                stats["total_downloaded"] += result["downloaded"]
                stats["by_breed"][breed] += result["downloaded"]
                print(f"  [{i:3d}/{len(breeds_this_round)}] {breed}: +{result['downloaded']} (total: {current_count + result['downloaded']})")

            # Rate limiting - vary delay to appear more human
            time.sleep(random.uniform(2, 5))

        stats["rounds_completed"] = round_num

        # Convert any webm files to mp3
        convert_webm_to_mp3()

        # Save stats after each round
        stats_file = OUTPUT_DIR / "download_stats.json"
        stats["last_update"] = datetime.now().isoformat()
        stats["total_files"] = sum(get_breed_file_count(b) for b in DOG_BREEDS)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        round_duration = (datetime.now() - round_start).total_seconds() / 60
        print(f"\nRound {round_num} complete: +{round_downloads} files in {round_duration:.1f} min")
        print(f"Total files: {stats['total_files']}")

        # Longer break between rounds
        if datetime.now() < end_time:
            sleep_time = random.uniform(30, 60)
            print(f"Sleeping {sleep_time:.0f}s before next round...")
            time.sleep(sleep_time)

    # Final conversion
    convert_webm_to_mp3()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_files = sum(get_breed_file_count(b) for b in DOG_BREEDS)
    print(f"Total runtime: {(datetime.now() - start_time).total_seconds() / 3600:.1f} hours")
    print(f"Rounds completed: {round_num}")
    print(f"Total audio files: {total_files}")

    print("\nTop 20 breeds by file count:")
    breed_counts = [(b, get_breed_file_count(b)) for b in DOG_BREEDS]
    breed_counts.sort(key=lambda x: -x[1])
    for breed, count in breed_counts[:20]:
        print(f"  {breed}: {count}")

    print("\nBreeds with fewest files:")
    for breed, count in breed_counts[-10:]:
        print(f"  {breed}: {count}")

    # Save final stats
    stats["end_time"] = datetime.now().isoformat()
    stats["total_files"] = total_files
    stats["breed_counts"] = {b: get_breed_file_count(b) for b in DOG_BREEDS}
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to: {stats_file}")
    print("=" * 70)
    print("DONE!")


if __name__ == "__main__":
    main()
