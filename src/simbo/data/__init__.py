"""
Simbo Data Module.

Contains databases and data structures for world metadata and other resources.
"""

from simbo.data.world_database import (
    WorldMetadata,
    EnvironmentType,
    Scale,
    TerrainType,
    SimulatorCompatibility,
    get_world_database,
    get_world_by_id,
    filter_worlds,
    search_worlds_by_text,
    WORLD_DATABASE,
)

__all__ = [
    "WorldMetadata",
    "EnvironmentType",
    "Scale",
    "TerrainType",
    "SimulatorCompatibility",
    "get_world_database",
    "get_world_by_id",
    "filter_worlds",
    "search_worlds_by_text",
    "WORLD_DATABASE",
]
