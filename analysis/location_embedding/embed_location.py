from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MODEL_ALIASES: Dict[str, str] = {
	"satclip": "satclip",
	"range": "range",
	"range+": "range",
	"geoclip": "geoclip",
	"alphaearth": "alphaearth",
	"alpha_earth": "alphaearth",
}


_REPO_DIR = Path(__file__).resolve().parent
_THIRD_PARTY_DIR = _REPO_DIR / "third_party"


def _ensure_third_party_on_syspath(*relative_parts: str) -> Path:
	"""Ensure a local third_party source path exists on sys.path and return it."""
	path = (_THIRD_PARTY_DIR / Path(*relative_parts)).resolve()
	if path.exists():
		path_str = str(path)
		if path_str not in sys.path:
			sys.path.insert(0, path_str)
	return path


def _normalize_model_name(name: str) -> str:
	key = name.strip().lower()
	if key not in MODEL_ALIASES:
		supported = ", ".join(sorted(set(MODEL_ALIASES.values())))
		raise ValueError(f"Unknown location embedder '{name}'. Supported: {supported}")
	return MODEL_ALIASES[key]


def _iter_batches(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
	for start in range(0, n, batch_size):
		end = min(start + batch_size, n)
		yield start, end


def _sanitize_latlon(latlon: np.ndarray) -> np.ndarray:
	arr = np.asarray(latlon, dtype=np.float64)
	if arr.ndim != 2 or arr.shape[1] != 2:
		raise ValueError("Coordinates must be an array with shape (N, 2) in [lat, lon] order")

	arr[:, 0] = np.nan_to_num(arr[:, 0], nan=0.0, posinf=90.0, neginf=-90.0)
	arr[:, 1] = np.nan_to_num(arr[:, 1], nan=0.0, posinf=180.0, neginf=-180.0)
	arr[:, 0] = np.clip(arr[:, 0], -90.0, 90.0)
	arr[:, 1] = np.clip(arr[:, 1], -180.0, 180.0)
	return arr


@dataclass
class EmbedderSpec:
	model_name: str
	device: str = "cpu"
	batch_size: int = 2048
	satclip_ckpt_path: Optional[str] = None
	range_db_path: Optional[str] = None
	range_model_name: str = "RANGE+"
	range_beta: float = 0.5
	alphaearth_year: int = 2024
	alphaearth_scale_meters: int = 10
	alphaearth_project: Optional[str] = 'metabarcoding-491221'


class BaseLocationEmbedder:
	def __init__(self, device: str = "cpu", batch_size: int = 2048) -> None:
		self.device = device
		self.batch_size = max(1, int(batch_size))

	@property
	def embedding_dim(self) -> int:
		raise NotImplementedError

	def encode(self, latlon: np.ndarray) -> np.ndarray:
		raise NotImplementedError


class GeoCLIPEmbedder(BaseLocationEmbedder):
	def __init__(self, device: str = "cpu", batch_size: int = 2048) -> None:
		super().__init__(device=device, batch_size=batch_size)
		try:
			import torch
			geoclip_mod = importlib.import_module("geoclip")
		except ImportError as exc:
			raise ImportError(
				"GeoCLIP backend requires 'geoclip' and 'torch'. Install with: pip install geoclip"
			) from exc

		self._torch = torch
		LocationEncoder = getattr(geoclip_mod, "LocationEncoder")
		self._model = LocationEncoder().double().to(self.device)
		self._model.eval()

	@property
	def embedding_dim(self) -> int:
		return 512

	def encode(self, latlon: np.ndarray) -> np.ndarray:
		coords = _sanitize_latlon(latlon)
		outs: List[np.ndarray] = []
		with self._torch.no_grad():
			for start, end in _iter_batches(len(coords), self.batch_size):
				batch = self._torch.tensor(coords[start:end], dtype=self._torch.float64, device=self.device)
				emb = self._model(batch).detach().cpu().numpy().astype(np.float32)
				outs.append(emb)
		return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.embedding_dim), dtype=np.float32)


class _RangeBackedEmbedder(BaseLocationEmbedder):
	def __init__(
		self,
		model_name: str,
		device: str,
		batch_size: int,
		satclip_ckpt_path: Optional[str],
		range_db_path: Optional[str],
		range_beta: float,
	) -> None:
		super().__init__(device=device, batch_size=batch_size)
		try:
			import torch
			try:
				load_model_mod = importlib.import_module("range.load_model")
			except ImportError:
				# Fallback to local source clone under analysis/location_embedding/third_party/RANGE
				_ensure_third_party_on_syspath("RANGE")
				load_model_mod = importlib.import_module("range.load_model")
		except ImportError as exc:
			raise ImportError(
				"RANGE backend import failed. Ensure RANGE source is available under "
				"analysis/location_embedding/third_party/RANGE and dependencies are installed. "
				f"Original error: {exc}"
			) from exc

		self._torch = torch
		load_model = getattr(load_model_mod, "load_model")

		ckpt_path = satclip_ckpt_path
		if ckpt_path is None:
			try:
				from huggingface_hub import hf_hub_download
			except ImportError as exc:
				raise ImportError(
					"RANGE backend needs huggingface-hub to fetch SatCLIP checkpoint when "
					"'satclip_ckpt_path' is not provided. Install with: pip install huggingface-hub"
				) from exc
			ckpt_path = hf_hub_download(
				repo_id="microsoft/SatCLIP-ResNet50-L10",
				filename="satclip-resnet50-l10.ckpt",
				repo_type="model",
			)

		kwargs = {}
		if "RANGE" in model_name.upper():
			if range_db_path is None:
				raise ValueError(
					"RANGE requires a precomputed database file path via 'range_db_path'."
				)
			kwargs["db_path"] = range_db_path
			kwargs["beta"] = float(range_beta)

		self._model = load_model(
			model_name=model_name,
			pretrained_path=ckpt_path,
			device=device,
			**kwargs,
		)
		self._model.eval()

		if hasattr(self._model, "location_feature_dim"):
			self._embedding_dim = int(getattr(self._model, "location_feature_dim"))
		elif "RANGE" in model_name.upper():
			self._embedding_dim = 1280
		else:
			self._embedding_dim = 256

	@property
	def embedding_dim(self) -> int:
		return self._embedding_dim

	def encode(self, latlon: np.ndarray) -> np.ndarray:
		coords_latlon = _sanitize_latlon(latlon)
		# RANGE codepaths expect [lon, lat]
		coords_lonlat = np.column_stack([coords_latlon[:, 1], coords_latlon[:, 0]])

		outs: List[np.ndarray] = []
		with self._torch.no_grad():
			for start, end in _iter_batches(len(coords_lonlat), self.batch_size):
				batch = self._torch.tensor(
					coords_lonlat[start:end], dtype=self._torch.float64, device=self.device
				)
				emb = self._model(batch)
				if hasattr(emb, "detach"):
					emb_np = emb.detach().cpu().numpy()
				else:
					emb_np = np.asarray(emb)
				outs.append(np.asarray(emb_np, dtype=np.float32))
		return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.embedding_dim), dtype=np.float32)


class SatCLIPEmbedder(BaseLocationEmbedder):
	def __init__(
		self,
		device: str = "cpu",
		batch_size: int = 2048,
		satclip_ckpt_path: Optional[str] = None,
	) -> None:
		super().__init__(device=device, batch_size=batch_size)
		try:
			import torch
			from huggingface_hub import hf_hub_download
		except ImportError as exc:
			raise ImportError(
				"SatCLIP backend requires 'torch' and 'huggingface-hub'. "
				"Install with: pip install torch huggingface-hub"
			) from exc

		# SatCLIP loaders exposed by the official repo:
		# - load_lightweight.py: get_satclip_loc_encoder (no lightning dependency)
		# - load.py: get_satclip
		get_satclip = None
		for mod_name in (
			"load_lightweight",
			"satclip.load_lightweight",
			"load",
			"satclip.load",
			"satclip.load_model",
		):
			try:
				mod = importlib.import_module(mod_name)
			except ImportError:
				continue
			candidate = getattr(mod, "get_satclip", None) or getattr(mod, "get_satclip_loc_encoder", None)
			if candidate is not None:
				get_satclip = candidate
				break

		if get_satclip is None:
			# Fallback to local source clone under analysis/location_embedding/third_party/satclip
			# Load exact file path to avoid ambiguous module-name imports.
			local_satclip_dir = _ensure_third_party_on_syspath("satclip", "satclip")
			for local_file, mod_alias in (
				("load_lightweight.py", "satclip_local_load_lightweight"),
				("load.py", "satclip_local_load"),
			):
				local_path = local_satclip_dir / local_file
				if not local_path.exists():
					continue
				try:
					spec = importlib.util.spec_from_file_location(mod_alias, str(local_path))
					if spec is not None and spec.loader is not None:
						mod = importlib.util.module_from_spec(spec)
						spec.loader.exec_module(mod)
						candidate = getattr(mod, "get_satclip", None) or getattr(mod, "get_satclip_loc_encoder", None)
						if candidate is not None:
							get_satclip = candidate
							break
				except Exception:
					continue

		if get_satclip is None:
			raise ImportError(
				"SatCLIP backend could not find a loader function. Expected one of "
				"'get_satclip' or 'get_satclip_loc_encoder' from SatCLIP source. "
				"If using local clone, ensure third_party/satclip is present and SatCLIP deps are installed."
			)

		ckpt_path = satclip_ckpt_path
		if ckpt_path is None:
			ckpt_path = hf_hub_download(
				repo_id="microsoft/SatCLIP-ResNet50-L10",
				filename="satclip-resnet50-l10.ckpt",
				repo_type="model",
			)

		self._torch = torch
		self._model = get_satclip(ckpt_path, device=device)
		self._model.eval()
		self._embedding_dim = int(getattr(self._model, "location_feature_dim", 256))

	@property
	def embedding_dim(self) -> int:
		return self._embedding_dim

	def encode(self, latlon: np.ndarray) -> np.ndarray:
		coords_latlon = _sanitize_latlon(latlon)
		# SatCLIP expects [lon, lat] coordinates (as in HF card examples)
		coords_lonlat = np.column_stack([coords_latlon[:, 1], coords_latlon[:, 0]])

		outs: List[np.ndarray] = []
		with self._torch.no_grad():
			for start, end in _iter_batches(len(coords_lonlat), self.batch_size):
				batch = self._torch.tensor(
					coords_lonlat[start:end], dtype=self._torch.float64, device=self.device
				)
				emb = self._model(batch)
				if hasattr(emb, "detach"):
					emb_np = emb.detach().cpu().numpy()
				else:
					emb_np = np.asarray(emb)
				outs.append(np.asarray(emb_np, dtype=np.float32))
		return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.embedding_dim), dtype=np.float32)


class RANGEEmbedder(_RangeBackedEmbedder):
	def __init__(
		self,
		device: str = "cpu",
		batch_size: int = 4096,
		satclip_ckpt_path: Optional[str] = None,
		range_db_path: Optional[str] = None,
		range_model_name: str = "RANGE+",
		range_beta: float = 0.5,
	) -> None:
		super().__init__(
			model_name=range_model_name,
			device=device,
			batch_size=batch_size,
			satclip_ckpt_path=satclip_ckpt_path,
			range_db_path=range_db_path,
			range_beta=range_beta,
		)


class AlphaEarthEmbedder(BaseLocationEmbedder):
	DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

	def __init__(
		self,
		device: str = "cpu",
		batch_size: int = 256,
		year: int = 2024,
		scale_meters: int = 10,
		project: Optional[str] = 'metabarcoding-vector',
	) -> None:
		super().__init__(device=device, batch_size=batch_size)
		try:
			ee = importlib.import_module("ee")
		except ImportError as exc:
			raise ImportError(
				"AlphaEarth backend requires Earth Engine API. Install with: pip install earthengine-api"
			) from exc

		self._ee = ee
		self.year = int(year)
		self.scale_meters = int(scale_meters)

		# Allow project override from env for cluster jobs.
		project = (
			project
			or os.environ.get("ALPHAEARTH_EE_PROJECT")
			or os.environ.get("EE_PROJECT")
		)

		# Optional service-account credentials path for non-interactive auth.
		# Preferred on clusters where browser-based auth is unavailable.
		credentials_path = (
			os.environ.get("ALPHAEARTH_EE_CREDENTIALS_JSON")
			or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
		)

		last_exc: Optional[Exception] = None

		if credentials_path:
			if not os.path.exists(credentials_path):
				raise RuntimeError(
					"Earth Engine credentials file not found. "
					"Set ALPHAEARTH_EE_CREDENTIALS_JSON (or GOOGLE_APPLICATION_CREDENTIALS) "
					f"to a valid JSON key file path. Got: {credentials_path}"
				)

			try:
				with open(credentials_path, "r", encoding="utf-8") as f:
					service_account_email = json.load(f).get("client_email")
			except Exception as exc:
				raise RuntimeError(
					"Could not read service-account JSON key for Earth Engine initialization. "
					f"Path: {credentials_path}"
				) from exc

			if not service_account_email:
				raise RuntimeError(
					"Service-account JSON is missing 'client_email'. "
					f"Path: {credentials_path}"
				)

			try:
				credentials = ee.ServiceAccountCredentials(service_account_email, credentials_path)
				ee.Initialize(credentials=credentials, project=project) if project else ee.Initialize(credentials=credentials)
				last_exc = None
			except Exception as exc:
				last_exc = exc

		if last_exc is not None or not credentials_path:
			try:
				ee.Initialize(project=project) if project else ee.Initialize()
			except Exception as exc:
				root = last_exc if last_exc is not None else exc
				raise RuntimeError(
					"Earth Engine is not initialized. "
					"For local interactive auth, run `earthengine authenticate`. "
					"For clusters, set ALPHAEARTH_EE_CREDENTIALS_JSON (or GOOGLE_APPLICATION_CREDENTIALS) "
					"to a service-account JSON key and set ALPHAEARTH_EE_PROJECT to your GCP project ID. "
					f"Initialization error: {root}"
				) from exc

	@property
	def embedding_dim(self) -> int:
		return 64

	def _encode_batch(self, latlon_batch: np.ndarray) -> np.ndarray:
		ee = self._ee
		features = []
		for i, (lat, lon) in enumerate(latlon_batch):
			geom = ee.Geometry.Point([float(lon), float(lat)])
			features.append(ee.Feature(geom, {"idx": int(i)}))

		start = f"{self.year}-01-01"
		end = f"{self.year + 1}-01-01"
		collection = ee.ImageCollection(self.DATASET_ID).filterDate(start, end)
		image = collection.mosaic()

		sampled = image.sampleRegions(
			collection=ee.FeatureCollection(features),
			properties=["idx"],
			scale=self.scale_meters,
			geometries=False,
		).getInfo()

		out = np.zeros((len(latlon_batch), self.embedding_dim), dtype=np.float32)
		for feature in sampled.get("features", []):
			props = feature.get("properties", {})
			idx = int(props.get("idx", -1))
			if idx < 0 or idx >= len(latlon_batch):
				continue
			out[idx] = np.array([props.get(f"A{i:02d}", 0.0) for i in range(64)], dtype=np.float32)
		return out

	def encode(self, latlon: np.ndarray) -> np.ndarray:
		coords = _sanitize_latlon(latlon)
		outs: List[np.ndarray] = []
		for start, end in _iter_batches(len(coords), self.batch_size):
			outs.append(self._encode_batch(coords[start:end]))
		return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.embedding_dim), dtype=np.float32)


def build_location_embedder(spec: EmbedderSpec) -> BaseLocationEmbedder:
	model_name = _normalize_model_name(spec.model_name)
	if model_name == "satclip":
		return SatCLIPEmbedder(
			device=spec.device,
			batch_size=spec.batch_size,
			satclip_ckpt_path=spec.satclip_ckpt_path,
		)
	if model_name == "range":
		return RANGEEmbedder(
			device=spec.device,
			batch_size=spec.batch_size,
			satclip_ckpt_path=spec.satclip_ckpt_path,
			range_db_path=spec.range_db_path,
			range_model_name=spec.range_model_name,
			range_beta=spec.range_beta,
		)
	if model_name == "geoclip":
		return GeoCLIPEmbedder(device=spec.device, batch_size=spec.batch_size)
	if model_name == "alphaearth":
		return AlphaEarthEmbedder(
			device=spec.device,
			batch_size=spec.batch_size,
			year=spec.alphaearth_year,
			scale_meters=spec.alphaearth_scale_meters,
			project=spec.alphaearth_project,
		)
	raise ValueError(f"Unsupported model_name '{spec.model_name}'")


def add_location_embeddings(
	df: pd.DataFrame,
	embedder: BaseLocationEmbedder,
	*,
	lat_col: str = "latitude",
	lon_col: str = "longitude",
	prefix: str = "loc_emb",
) -> Tuple[pd.DataFrame, List[str]]:
	if lat_col not in df.columns or lon_col not in df.columns:
		raise ValueError(f"DataFrame must contain '{lat_col}' and '{lon_col}' columns")

	coord_df = df[[lat_col, lon_col]].copy()
	valid_mask = coord_df[lat_col].notna() & coord_df[lon_col].notna()

	unique_coords = coord_df.loc[valid_mask, [lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
	emb_cols = [f"{prefix}_{i:03d}" for i in range(embedder.embedding_dim)]

	if len(unique_coords) == 0:
		out_df = df.copy()
		empty_emb_df = pd.DataFrame(
			np.zeros((len(out_df), len(emb_cols)), dtype=np.float32),
			columns=emb_cols,
			index=out_df.index,
		)
		out_df = pd.concat([out_df, empty_emb_df], axis=1)
		return out_df, emb_cols

	unique_latlon = unique_coords[[lat_col, lon_col]].to_numpy(dtype=np.float64)
	unique_embeddings = embedder.encode(unique_latlon)
	if unique_embeddings.shape[1] != embedder.embedding_dim:
		raise ValueError(
			"Embedding dimension mismatch: "
			f"expected {embedder.embedding_dim}, got {unique_embeddings.shape[1]}"
		)

	emb_values_df = pd.DataFrame(unique_embeddings, columns=emb_cols, index=unique_coords.index)
	emb_df = pd.concat([unique_coords, emb_values_df], axis=1)

	merged = df.merge(emb_df, on=[lat_col, lon_col], how="left")
	merged[emb_cols] = merged[emb_cols].fillna(0.0)
	return merged, emb_cols
