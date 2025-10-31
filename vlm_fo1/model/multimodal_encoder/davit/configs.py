
model_configs = {
    "davit-base": {
        "depths": [
            1,
            1,
            9,
            1
        ],
        "dim_embed": [
            128,
            256,
            512,
            1024
        ],
        "drop_path_rate": 0.1,
        "enable_checkpoint": True,
        "image_feature_source": [
            "spatial_avg_pool",
            "temporal_avg_pool"
        ],
        "image_pos_embed": {
            "max_pos_embeddings": 50,
            "type": "learned_abs_2d"
        },
        "num_groups": [
            4,
            8,
            16,
            32
        ],
        "num_heads": [
            4,
            8,
            16,
            32
        ],
        "patch_padding": [
            3,
            1,
            1,
            1
        ],
        "patch_prenorm": [
            False,
            True,
            True,
            True
        ],
        "patch_size": [
            7,
            3,
            3,
            3
        ],
        "patch_stride": [
            4,
            2,
            2,
            2
        ],
        "projection_dim": 768,
        "transformers_version": "4.41.2",
        "visual_temporal_embedding": {
            "max_temporal_embeddings": 100,
            "type": "COSINE"
        },
        "window_size": 12
    },
    "davit-large": {
        "depths": [
            1,
            1,
            9,
            1
        ],
        "dim_embed": [
            256,
            512,
            1024,
            2048
        ],
        "drop_path_rate": 0.1,
        "enable_checkpoint": True,
        "image_feature_source": [
            "spatial_avg_pool",
            "temporal_avg_pool"
        ],
        "image_pos_embed": {
            "max_pos_embeddings": 50,
            "type": "learned_abs_2d"
        },
        "num_groups": [
            8,
            16,
            32,
            64
        ],
        "num_heads": [
            8,
            16,
            32,
            64
        ],
        "patch_padding": [
            3,
            1,
            1,
            1
        ],
        "patch_prenorm": [
            False,
            True,
            True,
            True
        ],
        "patch_size": [
            7,
            3,
            3,
            3
        ],
        "patch_stride": [
            4,
            2,
            2,
            2
        ],
        "projection_dim": 1024,
        "transformers_version": "4.41.2",
        "visual_temporal_embedding": {
            "max_temporal_embeddings": 100,
            "type": "COSINE"
        },
        "window_size": 12
    }
}

img_cfg = {
    "do_resize": True,
    "size": {
        "height": 768,
        "width":768 
    },
    "resample": 3,
    "do_center_crop": False,
    "do_rescale": True,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "do_convert_rgb": True
}
