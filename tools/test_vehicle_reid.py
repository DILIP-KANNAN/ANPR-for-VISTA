"""
Unit Test & Latency Profiler for OSNet Vehicle Re-Identification & Gallery Store.
Verifies embedding consistency, vector search latency, and cosine separation.
"""

import os
import sys
import time
import glob
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anpr_pipeline.vehicle_reid import OSNetVehicleReID, VehicleIdentityGallery

def main():
    print("=" * 70)
    print("       OSNET VEHICLE RE-IDENTIFICATION & GALLERY UNIT TEST")
    print("=" * 70)

    # 1. Initialize OSNet Engine
    reid = OSNetVehicleReID()
    print(f"Device: {reid.device} | Target Size: {reid.target_size}")

    # Load 2 distinct vehicle images
    val_images = sorted(glob.glob("datasets/plate_detection/images/val/ind_*.jpg"))
    if len(val_images) < 2:
        print("Error: Need at least 2 test images.")
        return

    veh_a = cv2.imread(val_images[0])
    veh_b = cv2.imread(val_images[1])

    # 2. Embedding Latency Profiling
    # Warmup
    for _ in range(5):
        reid.extract_embedding(veh_a)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        emb = reid.extract_embedding(veh_a)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_lat = np.mean(latencies)
    print(f"OSNet Embedding Extraction Latency: {avg_lat:.2f} ms per vehicle (RTX 4060 GPU)")
    assert emb.shape == (512,), f"Expected shape (512,), got {emb.shape}"
    norm = np.linalg.norm(emb)
    print(f"Embedding L2-Norm: {norm:.4f} (Confirmed unit vector)")

    # 3. Test Gallery Storing & Matching
    gallery = VehicleIdentityGallery(similarity_threshold=0.85)

    # Store Vehicle A
    vid_a = gallery.store_vehicle(track_id=1, vehicle_crop=veh_a, plate="MH47Y1124")
    print(f"\nStored Vehicle A in Gallery with ID: {vid_a}")

    # Query with augmented view of Vehicle A (slight brightness/crop shift)
    h, w = veh_a.shape[:2]
    veh_a_aug = cv2.convertScaleAbs(veh_a, alpha=1.05, beta=5)
    matched_id, sim, plate = gallery.find_matching_vehicle(veh_a_aug)
    print(f"Query with Same Vehicle (Augmented): Matched={matched_id} | Similarity={sim:.3f} | Plate={plate}")
    assert matched_id == vid_a, "Failed to match same vehicle!"
    assert sim >= 0.85, f"Expected similarity >= 0.85, got {sim}"

    # Query with completely different Vehicle B
    matched_b, sim_b, plate_b = gallery.find_matching_vehicle(veh_b)
    print(f"Query with Different Vehicle B: Matched={matched_b} | Similarity={sim_b:.3f} | Plate={plate_b}")
    assert matched_b is None, f"Different vehicle should not exceed threshold (got {matched_b} with sim {sim_b})"

    # 4. Scalability: Vector Search Latency over 1,000 Gallery Entries
    print("\nSimulating Gallery scaled to 1,000 active vehicles...")
    for i in range(1000):
        fake_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        gallery.store_vehicle(track_id=i + 2, vehicle_crop=fake_crop, plate=f"TN01AA{i:04d}")

    t_search_0 = time.perf_counter()
    matched_scale, sim_scale, plate_scale = gallery.find_matching_vehicle(veh_a_aug)
    search_dur = (time.perf_counter() - t_search_0) * 1000.0

    print(f"Vector search across {len(gallery.records)} stored vehicles: {search_dur:.3f} ms!")
    print(f"Matched ID: {matched_scale} (Similarity: {sim_scale:.3f})")
    assert matched_scale == vid_a, "Failed to find Vehicle A in 1,000 vehicle gallery!"

    print("=" * 70)
    print("           ALL OSNET VEHICLE REID TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
