#!/usr/bin/env python3
"""
youtube_api.py – Thin helper around the YouTube Data v3 endpoints we need.
"""

from __future__ import annotations
import csv, pathlib
from typing import Iterable, List, Dict, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def yt_build(api_key: str):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


# ------------- basic retrieval helpers ----------------- #

def search_videos(service, query: str, max_results: int = 20, region: str | None = None) -> List[str]:
    resp = service.search().list(
        q=query, part="id", maxResults=min(max_results, 50),
        type="video", videoDuration="medium", videoEmbeddable="true",
        regionCode=region,
    ).execute()
    return [item["id"]["videoId"] for item in resp.get("items", [])]


def playlist_items(service, playlist_id: str, limit: int = 500) -> List[str]:
    vids, page = [], None
    while True:
        part = service.playlistItems().list(
            playlistId=playlist_id, part="contentDetails",
            maxResults=50, pageToken=page).execute()
        vids += [i["contentDetails"]["videoId"] for i in part.get("items", [])]
        page = part.get("nextPageToken")
        if not page or len(vids) >= limit:
            break
    return vids[:limit]


def channel_latest_playlists(service, channel_id: str, limit: int = 10) -> List[str]:
    try:
        resp = service.playlists().list(channelId=channel_id, part="id", maxResults=limit).execute()
    except HttpError:
        return []
    return [i["id"] for i in resp.get("items", [])]


def video_details(service, ids: List[str]) -> Dict[str, dict]:
    info: list[dict] = []
    for i in range(0, len(ids), 50):
        info += service.videos().list(
            part="snippet,contentDetails", id=",".join(ids[i:i + 50])
        ).execute().get("items", [])
    return {item["id"]: item for item in info}


# ------------- discovery from CSV / TXT ---------------- #

def iter_sources_csv(service, csv_path: str | pathlib.Path, max_per_playlist: int = 200) -> Iterable[str]:
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            kind, id_ = row["kind"], row["id"]
            if kind == "playlist":
                yield from playlist_items(service, id_, limit=max_per_playlist)
            elif kind == "channel":
                for pid in channel_latest_playlists(service, id_):
                    yield from playlist_items(service, pid, limit=max_per_playlist)
