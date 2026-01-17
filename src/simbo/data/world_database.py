"""
World Database - Metadata for open-source Gazebo/Ignition worlds.

This module contains a curated database of open-source world files with
structured metadata for semantic search and filtering.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class EnvironmentType(Enum):
    """Environment type classification."""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    MIXED = "mixed"


class Scale(Enum):
    """World scale classification."""
    SMALL = "small"      # < 50m x 50m
    MEDIUM = "medium"    # 50-200m x 50-200m
    LARGE = "large"      # > 200m x 200m


class TerrainType(Enum):
    """Terrain type classification."""
    FLAT = "flat"
    UNEVEN = "uneven"
    HEIGHTMAP = "heightmap"
    MULTI_LEVEL = "multi_level"


class SimulatorCompatibility(Enum):
    """Simulator compatibility."""
    GAZEBO_CLASSIC = "gazebo_classic"  # Gazebo 9/11
    IGNITION = "ignition"              # Ignition Gazebo (Fortress, Garden, etc.)
    BOTH = "both"


@dataclass
class WorldMetadata:
    """Metadata for a Gazebo/Ignition world file."""

    # Identifiers
    id: str
    name: str
    description: str

    # Source information
    source_repo: str
    source_url: str
    world_file_path: str  # Path within the repo
    license: str

    # Classification
    environment_type: EnvironmentType
    scale: Scale
    terrain_type: TerrainType
    simulator_compatibility: SimulatorCompatibility

    # Semantic features (tags)
    semantic_features: List[str] = field(default_factory=list)

    # Technical details
    has_models: bool = True
    has_lights: bool = True
    has_physics: bool = True
    complexity_score: int = 5  # 1-10 scale

    # Embedding text (for vector search)
    embedding_text: str = ""

    def __post_init__(self):
        """Generate embedding text from metadata."""
        if not self.embedding_text:
            self.embedding_text = self._generate_embedding_text()

    def _generate_embedding_text(self) -> str:
        """Generate text suitable for embedding."""
        parts = [
            self.name,
            self.description,
            f"{self.environment_type.value} environment",
            f"{self.scale.value} scale",
            f"{self.terrain_type.value} terrain",
            " ".join(self.semantic_features),
        ]
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source_repo": self.source_repo,
            "source_url": self.source_url,
            "world_file_path": self.world_file_path,
            "license": self.license,
            "environment_type": self.environment_type.value,
            "scale": self.scale.value,
            "terrain_type": self.terrain_type.value,
            "simulator_compatibility": self.simulator_compatibility.value,
            "semantic_features": self.semantic_features,
            "has_models": self.has_models,
            "has_lights": self.has_lights,
            "has_physics": self.has_physics,
            "complexity_score": self.complexity_score,
            "embedding_text": self.embedding_text,
        }


# =============================================================================
# WORLD DATABASE
# =============================================================================

WORLD_DATABASE: List[WorldMetadata] = [
    # -------------------------------------------------------------------------
    # Indoor Worlds
    # -------------------------------------------------------------------------
    WorldMetadata(
        id="aws_robomaker_small_house",
        name="AWS Small House World",
        description="A small residential house with multiple rooms including living room, kitchen, bedroom, and bathroom. Ideal for indoor navigation and domestic robot testing.",
        source_repo="aws-robotics/aws-robomaker-small-house-world",
        source_url="https://github.com/aws-robotics/aws-robomaker-small-house-world",
        world_file_path="worlds/small_house.world",
        license="MIT",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.SMALL,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.GAZEBO_CLASSIC,
        semantic_features=["house", "home", "residential", "domestic", "rooms", "furniture", "living room", "kitchen", "bedroom", "bathroom"],
        complexity_score=6,
    ),

    WorldMetadata(
        id="aws_robomaker_hospital",
        name="AWS Hospital World",
        description="A hospital environment with corridors, patient rooms, waiting areas, and medical equipment. Suitable for healthcare robot simulation.",
        source_repo="aws-robotics/aws-robomaker-hospital-world",
        source_url="https://github.com/aws-robotics/aws-robomaker-hospital-world",
        world_file_path="worlds/hospital.world",
        license="MIT",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.MEDIUM,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.GAZEBO_CLASSIC,
        semantic_features=["hospital", "medical", "healthcare", "corridors", "patient rooms", "waiting area", "clinic"],
        complexity_score=7,
    ),

    WorldMetadata(
        id="aws_robomaker_bookstore",
        name="AWS Bookstore World",
        description="A bookstore environment with bookshelves, aisles, and checkout counters. Good for retail robot testing and navigation.",
        source_repo="aws-robotics/aws-robomaker-bookstore-world",
        source_url="https://github.com/aws-robotics/aws-robomaker-bookstore-world",
        world_file_path="worlds/bookstore.world",
        license="MIT",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.SMALL,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.GAZEBO_CLASSIC,
        semantic_features=["bookstore", "retail", "store", "shop", "shelves", "aisles", "commercial"],
        complexity_score=5,
    ),

    WorldMetadata(
        id="turtlebot3_house",
        name="TurtleBot3 House World",
        description="A simple house environment designed for TurtleBot3 navigation. Contains basic furniture and room layouts.",
        source_repo="ROBOTIS-GIT/turtlebot3_simulations",
        source_url="https://github.com/ROBOTIS-GIT/turtlebot3_simulations",
        world_file_path="turtlebot3_gazebo/worlds/turtlebot3_house.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.SMALL,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.GAZEBO_CLASSIC,
        semantic_features=["house", "home", "turtlebot", "navigation", "simple", "basic", "furniture"],
        complexity_score=3,
    ),

    WorldMetadata(
        id="turtlebot3_world",
        name="TurtleBot3 World",
        description="A maze-like environment with walls and obstacles for TurtleBot3 SLAM and navigation testing.",
        source_repo="ROBOTIS-GIT/turtlebot3_simulations",
        source_url="https://github.com/ROBOTIS-GIT/turtlebot3_simulations",
        world_file_path="turtlebot3_gazebo/worlds/turtlebot3_world.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.SMALL,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.GAZEBO_CLASSIC,
        semantic_features=["maze", "obstacles", "walls", "turtlebot", "slam", "navigation", "test"],
        complexity_score=2,
    ),

    WorldMetadata(
        id="gazebo_warehouse",
        name="Gazebo Warehouse",
        description="A warehouse environment with shelving units, pallets, and open spaces. Ideal for logistics and warehouse robot simulation.",
        source_repo="osrf/gazebo_models",
        source_url="https://github.com/osrf/gazebo_models",
        world_file_path="worlds/warehouse.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.MEDIUM,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.BOTH,
        semantic_features=["warehouse", "logistics", "shelves", "pallets", "storage", "industrial", "racks", "forklift"],
        complexity_score=6,
    ),

    WorldMetadata(
        id="gazebo_cafe",
        name="Gazebo Cafe World",
        description="A cafe environment with tables, chairs, and counter. Good for service robot simulation.",
        source_repo="osrf/gazebo_models",
        source_url="https://github.com/osrf/gazebo_models",
        world_file_path="worlds/cafe.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.SMALL,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.BOTH,
        semantic_features=["cafe", "restaurant", "tables", "chairs", "service", "food", "counter"],
        complexity_score=4,
    ),

    WorldMetadata(
        id="office_world",
        name="Office Environment",
        description="A typical office environment with desks, cubicles, meeting rooms, and corridors. Suitable for office robot applications.",
        source_repo="osrf/gazebo_models",
        source_url="https://github.com/osrf/gazebo_models",
        world_file_path="worlds/office.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.MEDIUM,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.BOTH,
        semantic_features=["office", "workplace", "desks", "cubicles", "meeting room", "corridors", "corporate"],
        complexity_score=5,
    ),

    WorldMetadata(
        id="factory_floor",
        name="Factory Floor World",
        description="An industrial factory floor with conveyor belts, workstations, and heavy machinery. For industrial robot simulation.",
        source_repo="osrf/gazebo_models",
        source_url="https://github.com/osrf/gazebo_models",
        world_file_path="worlds/factory.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.BOTH,
        semantic_features=["factory", "industrial", "manufacturing", "conveyor", "machinery", "production", "assembly"],
        complexity_score=8,
    ),

    # -------------------------------------------------------------------------
    # Outdoor Worlds
    # -------------------------------------------------------------------------
    WorldMetadata(
        id="clearpath_office_outdoor",
        name="Clearpath Office Outdoor",
        description="Outdoor environment around an office building with parking lot, sidewalks, and landscaping.",
        source_repo="clearpathrobotics/cpr_gazebo",
        source_url="https://github.com/clearpathrobotics/cpr_gazebo",
        world_file_path="cpr_office_gazebo/worlds/office.world",
        license="BSD-3-Clause",
        environment_type=EnvironmentType.OUTDOOR,
        scale=Scale.MEDIUM,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.GAZEBO_CLASSIC,
        semantic_features=["outdoor", "parking", "sidewalk", "office", "urban", "landscaping", "building"],
        complexity_score=5,
    ),

    WorldMetadata(
        id="sonoma_raceway",
        name="Sonoma Raceway",
        description="A detailed racing circuit with track, barriers, and pit areas. For autonomous vehicle testing.",
        source_repo="osrf/car_demo",
        source_url="https://github.com/osrf/car_demo",
        world_file_path="worlds/sonoma_raceway.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.OUTDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.GAZEBO_CLASSIC,
        semantic_features=["racing", "track", "circuit", "autonomous vehicle", "car", "road", "driving"],
        complexity_score=7,
    ),

    WorldMetadata(
        id="citysim",
        name="CitySim Urban World",
        description="A realistic urban city environment with roads, buildings, traffic lights, and intersections. Perfect for autonomous driving simulation.",
        source_repo="osrf/citysim",
        source_url="https://github.com/osrf/citysim",
        world_file_path="worlds/city.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.OUTDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.GAZEBO_CLASSIC,
        semantic_features=["city", "urban", "roads", "streets", "buildings", "traffic lights", "intersections", "autonomous driving"],
        complexity_score=9,
    ),

    WorldMetadata(
        id="gazebo_willowgarage",
        name="Willow Garage Office",
        description="The famous Willow Garage office environment. A classic world for PR2 and mobile robot testing.",
        source_repo="osrf/gazebo_models",
        source_url="https://github.com/osrf/gazebo_models",
        world_file_path="worlds/willowgarage.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.BOTH,
        semantic_features=["office", "willow garage", "pr2", "classic", "large", "corridors", "rooms"],
        complexity_score=7,
    ),

    WorldMetadata(
        id="clearpath_agriculture",
        name="Clearpath Agriculture Field",
        description="An agricultural field environment with crop rows and farm structures. For agricultural robot simulation.",
        source_repo="clearpathrobotics/cpr_gazebo",
        source_url="https://github.com/clearpathrobotics/cpr_gazebo",
        world_file_path="cpr_agriculture_gazebo/worlds/agriculture.world",
        license="BSD-3-Clause",
        environment_type=EnvironmentType.OUTDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.UNEVEN,
        simulator_compatibility=SimulatorCompatibility.GAZEBO_CLASSIC,
        semantic_features=["agriculture", "farm", "field", "crops", "rural", "farming", "outdoor"],
        complexity_score=5,
    ),

    WorldMetadata(
        id="clearpath_inspection",
        name="Clearpath Inspection World",
        description="An industrial inspection environment with pipes, tanks, and catwalks. For inspection robot testing.",
        source_repo="clearpathrobotics/cpr_gazebo",
        source_url="https://github.com/clearpathrobotics/cpr_gazebo",
        world_file_path="cpr_inspection_gazebo/worlds/inspection.world",
        license="BSD-3-Clause",
        environment_type=EnvironmentType.MIXED,
        scale=Scale.MEDIUM,
        terrain_type=TerrainType.MULTI_LEVEL,
        simulator_compatibility=SimulatorCompatibility.GAZEBO_CLASSIC,
        semantic_features=["inspection", "industrial", "pipes", "tanks", "catwalks", "plant", "maintenance"],
        complexity_score=6,
    ),

    WorldMetadata(
        id="forest_world",
        name="Forest Environment",
        description="A forest environment with trees, uneven terrain, and natural obstacles. For outdoor robot navigation.",
        source_repo="osrf/gazebo_models",
        source_url="https://github.com/osrf/gazebo_models",
        world_file_path="worlds/forest.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.OUTDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.UNEVEN,
        simulator_compatibility=SimulatorCompatibility.BOTH,
        semantic_features=["forest", "trees", "nature", "outdoor", "vegetation", "natural", "woods"],
        complexity_score=6,
    ),

    WorldMetadata(
        id="lunar_surface",
        name="Lunar Surface World",
        description="A lunar surface environment with craters and rocky terrain. For space robot simulation.",
        source_repo="osrf/gazebo_models",
        source_url="https://github.com/osrf/gazebo_models",
        world_file_path="worlds/moon.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.OUTDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.HEIGHTMAP,
        simulator_compatibility=SimulatorCompatibility.BOTH,
        semantic_features=["lunar", "moon", "space", "craters", "rocky", "extraterrestrial", "rover"],
        complexity_score=5,
    ),

    WorldMetadata(
        id="mars_terrain",
        name="Mars Terrain World",
        description="A Mars-like terrain with red rocky surface and craters. For planetary rover simulation.",
        source_repo="osrf/gazebo_models",
        source_url="https://github.com/osrf/gazebo_models",
        world_file_path="worlds/mars.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.OUTDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.HEIGHTMAP,
        simulator_compatibility=SimulatorCompatibility.BOTH,
        semantic_features=["mars", "planetary", "rover", "space", "rocky", "red", "extraterrestrial"],
        complexity_score=5,
    ),

    # -------------------------------------------------------------------------
    # Ignition Gazebo Worlds
    # -------------------------------------------------------------------------
    WorldMetadata(
        id="ign_fuel_depot",
        name="Ignition Fuel Depot",
        description="A fuel depot environment for Ignition Gazebo with tanks, pipes, and industrial structures.",
        source_repo="gazebosim/gz-sim",
        source_url="https://github.com/gazebosim/gz-sim",
        world_file_path="examples/worlds/fuel_depot.sdf",
        license="Apache-2.0",
        environment_type=EnvironmentType.OUTDOOR,
        scale=Scale.MEDIUM,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.IGNITION,
        semantic_features=["fuel", "depot", "industrial", "tanks", "ignition", "modern"],
        complexity_score=6,
    ),

    WorldMetadata(
        id="ign_tunnel",
        name="Ignition Tunnel World",
        description="A tunnel environment for Ignition Gazebo. Suitable for underground robot testing.",
        source_repo="gazebosim/gz-sim",
        source_url="https://github.com/gazebosim/gz-sim",
        world_file_path="examples/worlds/tunnel.sdf",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.MEDIUM,
        terrain_type=TerrainType.UNEVEN,
        simulator_compatibility=SimulatorCompatibility.IGNITION,
        semantic_features=["tunnel", "underground", "cave", "mining", "dark", "ignition"],
        complexity_score=5,
    ),

    WorldMetadata(
        id="ign_shapes",
        name="Ignition Shapes World",
        description="A simple world with basic geometric shapes for testing. Good for quick debugging.",
        source_repo="gazebosim/gz-sim",
        source_url="https://github.com/gazebosim/gz-sim",
        world_file_path="examples/worlds/shapes.sdf",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.SMALL,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.IGNITION,
        semantic_features=["shapes", "simple", "basic", "test", "debug", "ignition", "geometric"],
        complexity_score=1,
    ),

    # -------------------------------------------------------------------------
    # SubT Challenge Worlds
    # -------------------------------------------------------------------------
    WorldMetadata(
        id="subt_urban",
        name="SubT Urban World",
        description="Urban underground environment from the DARPA SubT Challenge. Multi-level with corridors and rooms.",
        source_repo="osrf/subt",
        source_url="https://github.com/osrf/subt",
        world_file_path="subt_ign/worlds/urban_circuit.sdf",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.MULTI_LEVEL,
        simulator_compatibility=SimulatorCompatibility.IGNITION,
        semantic_features=["subt", "underground", "urban", "multi-level", "corridors", "challenge", "exploration"],
        complexity_score=9,
    ),

    WorldMetadata(
        id="subt_cave",
        name="SubT Cave World",
        description="Natural cave environment from the DARPA SubT Challenge. Uneven terrain with narrow passages.",
        source_repo="osrf/subt",
        source_url="https://github.com/osrf/subt",
        world_file_path="subt_ign/worlds/cave_circuit.sdf",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.UNEVEN,
        simulator_compatibility=SimulatorCompatibility.IGNITION,
        semantic_features=["subt", "cave", "underground", "natural", "narrow", "exploration", "challenge"],
        complexity_score=8,
    ),

    WorldMetadata(
        id="subt_tunnel",
        name="SubT Tunnel World",
        description="Tunnel environment from the DARPA SubT Challenge. Long corridors with mining infrastructure.",
        source_repo="osrf/subt",
        source_url="https://github.com/osrf/subt",
        world_file_path="subt_ign/worlds/tunnel_circuit.sdf",
        license="Apache-2.0",
        environment_type=EnvironmentType.INDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.IGNITION,
        semantic_features=["subt", "tunnel", "mining", "underground", "corridors", "exploration", "challenge"],
        complexity_score=8,
    ),

    # -------------------------------------------------------------------------
    # Simple/Basic Worlds
    # -------------------------------------------------------------------------
    WorldMetadata(
        id="empty_world",
        name="Empty World",
        description="A minimal empty world with just a ground plane. Perfect starting point for custom scenarios.",
        source_repo="osrf/gazebo",
        source_url="https://github.com/gazebosim/gazebo-classic",
        world_file_path="worlds/empty.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.OUTDOOR,
        scale=Scale.LARGE,
        terrain_type=TerrainType.FLAT,
        simulator_compatibility=SimulatorCompatibility.BOTH,
        semantic_features=["empty", "basic", "simple", "minimal", "ground", "blank", "starting point"],
        complexity_score=1,
    ),

    WorldMetadata(
        id="rubble_world",
        name="Rubble World",
        description="A disaster scene with rubble and debris. For search and rescue robot simulation.",
        source_repo="osrf/gazebo_models",
        source_url="https://github.com/osrf/gazebo_models",
        world_file_path="worlds/rubble.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.OUTDOOR,
        scale=Scale.MEDIUM,
        terrain_type=TerrainType.UNEVEN,
        simulator_compatibility=SimulatorCompatibility.BOTH,
        semantic_features=["rubble", "disaster", "debris", "search and rescue", "emergency", "destruction"],
        complexity_score=6,
    ),

    WorldMetadata(
        id="heightmap_bowl",
        name="Heightmap Bowl World",
        description="A bowl-shaped terrain using heightmaps. Good for testing robots on slopes.",
        source_repo="osrf/gazebo",
        source_url="https://github.com/gazebosim/gazebo-classic",
        world_file_path="worlds/heightmap_bowl.world",
        license="Apache-2.0",
        environment_type=EnvironmentType.OUTDOOR,
        scale=Scale.MEDIUM,
        terrain_type=TerrainType.HEIGHTMAP,
        simulator_compatibility=SimulatorCompatibility.BOTH,
        semantic_features=["heightmap", "terrain", "bowl", "slopes", "elevation", "hills"],
        complexity_score=3,
    ),
]


def get_world_database() -> List[WorldMetadata]:
    """Get the complete world database."""
    return WORLD_DATABASE


def get_world_by_id(world_id: str) -> Optional[WorldMetadata]:
    """Get a specific world by its ID."""
    for world in WORLD_DATABASE:
        if world.id == world_id:
            return world
    return None


def filter_worlds(
    environment_type: Optional[EnvironmentType] = None,
    scale: Optional[Scale] = None,
    terrain_type: Optional[TerrainType] = None,
    simulator_compatibility: Optional[SimulatorCompatibility] = None,
    semantic_features: Optional[List[str]] = None,
    max_complexity: Optional[int] = None,
) -> List[WorldMetadata]:
    """
    Filter worlds based on constraints.

    Args:
        environment_type: Filter by environment type (indoor/outdoor/mixed)
        scale: Filter by scale (small/medium/large)
        terrain_type: Filter by terrain type (flat/uneven/heightmap/multi_level)
        simulator_compatibility: Filter by simulator (gazebo_classic/ignition/both)
        semantic_features: List of required semantic features (any match)
        max_complexity: Maximum complexity score (1-10)

    Returns:
        List of matching WorldMetadata objects
    """
    results = WORLD_DATABASE.copy()

    if environment_type is not None:
        results = [w for w in results if w.environment_type == environment_type]

    if scale is not None:
        results = [w for w in results if w.scale == scale]

    if terrain_type is not None:
        results = [w for w in results if w.terrain_type == terrain_type]

    if simulator_compatibility is not None:
        results = [w for w in results
                   if w.simulator_compatibility == simulator_compatibility
                   or w.simulator_compatibility == SimulatorCompatibility.BOTH]

    if semantic_features:
        # Match any of the provided features
        semantic_features_lower = [f.lower() for f in semantic_features]
        results = [
            w for w in results
            if any(
                any(sf in feature.lower() for feature in w.semantic_features)
                for sf in semantic_features_lower
            )
        ]

    if max_complexity is not None:
        results = [w for w in results if w.complexity_score <= max_complexity]

    return results


def search_worlds_by_text(query: str, limit: int = 10) -> List[WorldMetadata]:
    """
    Simple text-based search across world metadata.

    Args:
        query: Search query
        limit: Maximum number of results

    Returns:
        List of matching WorldMetadata objects sorted by relevance
    """
    query_lower = query.lower()
    query_terms = query_lower.split()

    scored_results = []

    for world in WORLD_DATABASE:
        score = 0
        searchable_text = (
            world.name.lower() + " " +
            world.description.lower() + " " +
            " ".join(world.semantic_features).lower() +
            world.environment_type.value + " " +
            world.terrain_type.value
        )

        # Score based on term matches
        for term in query_terms:
            if term in world.name.lower():
                score += 10  # Name match is highly relevant
            if term in world.description.lower():
                score += 5
            if any(term in feature.lower() for feature in world.semantic_features):
                score += 8  # Semantic feature match is very relevant
            if term in searchable_text:
                score += 1  # General match

        if score > 0:
            scored_results.append((world, score))

    # Sort by score descending
    scored_results.sort(key=lambda x: x[1], reverse=True)

    return [world for world, score in scored_results[:limit]]
