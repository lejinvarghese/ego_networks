from geopy.distance import geodesic


def geo_filter(
    anchor_coords: tuple[float, float],
    candidates: list[dict[str, float]],
    max_distance_km: float = 2,
) -> list[dict[str, float]]:
    filtered = []
    for p in candidates:
        distance = geodesic(anchor_coords, (p["lat"], p["lon"])).km
        if distance <= max_distance_km:
            p = p.copy()
            p["distance"] = round(distance, 4)
            filtered.append(p)
    return filtered


def greedy_path(start_place, candidates, max_hops=3, max_km=20):
    path = [start_place]
    current = start_place
    for _ in range(max_hops):
        options = geo_filter(
            (current["lat"], current["lon"]), candidates, max_km
        )
        options = [p for p in options if p["id"] not in [x["id"] for x in path]]
        if not options:
            break
        next_place = min(options, key=lambda x: x["distance"])
        path.append(next_place)
        current = next_place

    path_str = " -> ".join(p["name"] for p in path)

    return {
        "path": path,
        "formatted_path": path_str,
        "total_distance": sum(
            p.get("distance", 0) for p in path[1:]
        ),  # Skip start place
    }
