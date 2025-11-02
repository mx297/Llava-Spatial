def build_spatial_tower(spatial_tower_cfg,**kwargs):
    spatial_tower = getattr(spatial_tower_cfg, "spatial_tower", None)
    if spatial_tower == "vggt":
        # Use relative import for the encoder wrapper/adapter file
        from vggt import VGGTSpatialTower
        return VGGTSpatialTower(spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {spatial_tower}")