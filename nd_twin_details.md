ND_TWIN_Dataset. Images are face detected, face aligned, cutout face only, resize into 224x224.

Total images: 7025

The dataset infor are stored in 2 json file: id_to_images.json store paths to images; twin_pairs_infor.json store information on id twin pairs, splitted into train, validation and test. Train pairs are for training, val pairs are for validation, test pairs are for testing.

Images must be loaded from paths in id_to_images.json files.

Dataset details (There is a triplet, organize into 3 twin pairs):

Total: 204 twin pairs from 405 IDs. Total of 7025 images

Train: 160 twin pairs from 317 IDs, 5953 images

Test: 29 twin pairs from 58 IDs, 689 images

Val: 15 twin pairs from 30 IDs, 383 images

Total images: 7025
Train same person pairs: 85901
Train twin pairs: 88501
Train non-twin pairs: 35331842
Test same person pairs: 6534
Test twin pairs: 6804
Test non-twin pairs: 460964
Val same person pairs: 3488
Val twin pairs: 3677
Val non-twin pairs: 139330

id_to_images.json
```json
{
    "id1": [
        "/path/to/image_1/of/id1.jpg",
        "/path/to/image_2/of/id1.jpg",
        "/path/to/image_3/of/id1.jpg",
        "/path/to/image_4/of/id1.jpg"
    ],
    "id2": [
        "/path/to/image_1/of/id2.jpg",
        "/path/to/image_2/of/id2.jpg",
        "/path/to/image_3/of/id2.jpg",
        "/path/to/image_4/of/id2.jpg",
        "/path/to/image_5/of/id2.jpg"
    ],
    "id3": [
        "/path/to/image_1/of/id3.jpg",
        "/path/to/image_2/of/id3.jpg",
        "/path/to/image_3/of/id3.jpg",
        "/path/to/image_4/of/id3.jpg",
        "/path/to/image_5/of/id3.jpg",
        "/path/to/image_6/of/id3.jpg"
    ],
    "id4": [
        "/path/to/image_1/of/id4.jpg",
        "/path/to/image_2/of/id4.jpg",
        "/path/to/image_3/of/id4.jpg",
        "/path/to/image_4/of/id4.jpg"
    ]
}
```

twin_pairs_infor.json
```json
{
    "train" : [
        [
            "train_id1",
            "id_of_train_id1_twin"
        ],
        [
            "train_id2",
            "id_of_train_id2_twin"
        ]
    ],
    "test" : [
        [
            "test_id1",
            "id_of_test_id1_twin"
        ],
        [
            "test_id2",
            "id_of_test_id2_twin"
        ]
    ],
    "val" : [
        [
            "val_id1",
            "id_of_val_id1_twin"
        ],
        [
            "val_id2",
            "id_of_val_id2_twin"
        ]
    ]
}
```